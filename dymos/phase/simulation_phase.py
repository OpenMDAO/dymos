from .phase import Phase
from ..transcriptions import ExplicitShooting


class SimulationPhase(Phase):
    """
    A special phase class for performing simulation of an ODE without derivatives.

    Note: For the arguments which accept _unspecified, any argument not provided will be overridden
    by the simulation_options of the phase specified by `from_phase`.

    Parameters
    ----------
    transcription : ExplicitShooting
        The transcription used for the SimulationPhase. It must be an instance of ExplicitShooting.

    **kwargs : dict
        Dictionary of optional phase arguments.
    """
    def __init__(self, transcription=None, **kwargs):
        if not isinstance(transcription, ExplicitShooting):
            raise ValueError('The transcription for a SimulationPhase must be '
                             'ExplicitShooting. Use Phase.get_simulation_phase()'
                             'to create a simulation Phase.')
        super().__init__(transcription=transcription, **kwargs)

    def duplicate(self, *args, **kwargs):
        """
        Create a copy of this phase where most options and attributes are deep copies of those in the original.

        By default, a deepcopy of the transcription in the original phase is used.
        Boundary constraints, path constraints, and objectives are _NOT_ copied by default, but the user may opt to do so.
        By default, initial time is not fixed, nor are the initial or final state values.
        These also can be overridden with the appropriate arguments.

        Parameters
        ----------
        *args
            Additional arguments.
        **kwargs
            Keyword arguments.

        Raises
        ------
        NotImplmentedError
            This method is not yet supported for SimulationPhase
        """
        raise NotImplementedError('SimulationPhase does not support the duplicate method.')

    def set_vals_from_phase(self, from_phase):
        """
        Set the necessary values to simulate the phase based on variables in the given phase.

        Parameters
        ----------
        from_phase : Phase
            The dymos phase from which this simulation phase should pull its values.
        """
        # The use of `from_src=False` in the get_val calls here is due to the fact that the input/output
        # vectors are in `from_phase` are already populated and we don't need to track these values
        # to their ultimate source.

        t_initial = from_phase.get_val('t_initial', units=self.time_options['units'], from_src=False)
        self.set_val('t_initial', t_initial, units=self.time_options['units'])

        t_duration = from_phase.get_val('t_duration', units=self.time_options['units'], from_src=False)
        self.set_val('t_duration', t_duration, units=self.time_options['units'])

        avail_io = {meta['prom_name'] for meta in
                    from_phase.get_io_metadata(iotypes=('input', 'output'), get_remote=True).values()}

        for name, options in self.state_options.items():
            if f'states:{name}' in avail_io:
                val = from_phase.get_val(f'states:{name}', units=options['units'], from_src=False)[0, ...]
            elif f'initial_states:{name}' in avail_io:
                val = from_phase.get_val(f'initial_states:{name}', units=options['units'], from_src=False)
            else:
                raise RuntimeError('Unable to find state values in original phase')
            self.set_val(f'initial_states:{name}', val, units=options['units'])

        for name, options in self.parameter_options.items():
            val = from_phase.get_val(f'parameters:{name}', units=options['units'], from_src=False)
            self.set_val(f'parameters:{name}', val, units=options['units'])

        for name, options in self.control_options.items():
            val = from_phase.get_val(f'controls:{name}', units=options['units'], from_src=False)
            self.set_val(f'controls:{name}', val, units=options['units'])

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
        raise NotImplementedError('SimulationPhase does not support boundary constraints.')

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
        raise NotImplementedError('SimulationPhase does not support path constraints.')

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
        raise NotImplementedError('SimulationPhase does not support optimization objectives.')

    def check_time_options(self):
        """
        Check that time options are valid and issue warnings if invalid options are provided.

        This check is not performed by SimulationPhase.

        Warns
        -----
        RuntimeWarning
            RuntimeWarning is issued in the case of one or more invalid time options.
        """
        pass

    def _check_control_options(self):
        """
        Check that control options are valid and issue warnings if invalid options are provided.

        This check is not performed by SimulationPhase.

        Warns
        -----
        RuntimeWarning
            RuntimeWarning is issued in the case of one or more invalid time options.
        """
        pass

    def _check_parameter_options(self):
        """
        Check that parameter options are valid and issue warnings if invalid options are provided.

        This check is not performed by SimulationPhase.

        Warns
        -----
        RuntimeWarning
            RuntimeWarning is issued in the case of one or more invalid time options.
        """
        pass
