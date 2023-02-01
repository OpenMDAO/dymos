from openmdao.utils.mpi import MPI

from .phase import Phase
from ..transcriptions.grid_data import GaussLobattoGrid, RadauGrid, UniformGrid
from ..transcriptions import ExplicitShooting, GaussLobatto, Radau

from ..utils.misc import _unspecified


class SimulationPhase(Phase):
    """
    A special phase class for performing simulation of an ODE without derivatives.

    Note: For the arguments which accept _unspecified, any argument not provided will be overridden
    by the simulation_options of the phase specified by `from_phase`.

    Parameters
    ----------
    from_phase : <Phase> or None
        A phase instance from which the initialized phase should copy its data.
    times_per_seg : int
        The number of output points per segment, uniformly distributed.
    method : str or _unspecified
        A valid scipy.solve_ivp method for integration.
    atol : float or _unspecified
        Absolute error tolerance of the integration.
    rtol : float or _unspecified
        Relative error tolerance of the integration.
    first_step : float or _unspecified
        Initial step size of the integration.
    max_step : float or _unspecified
        Maximum step size of the integration.
    reports : float or _unspecified
        If True, generate reports for the subproblem used in integration.

    **kwargs : dict
        Dictionary of optional phase arguments.
    """
    def __init__(self, from_phase, times_per_seg=None, method=_unspecified, atol=_unspecified,
                 rtol=_unspecified, first_step=_unspecified, max_step=_unspecified,
                 reports=False, **kwargs):

        phase_tx = from_phase.options['transcription']
        num_seg = phase_tx.grid_data.num_segments
        seg_order = phase_tx.grid_data.transcription_order
        seg_ends = phase_tx.grid_data.segment_ends
        compressed = phase_tx.grid_data.compressed

        _method = method if method is not _unspecified else from_phase.simulate_options['method']
        _atol = atol if atol is not _unspecified else from_phase.simulate_options['atol']
        _rtol = rtol if rtol is not _unspecified else from_phase.simulate_options['rtol']
        _first_step = first_step if first_step is not _unspecified else from_phase.simulate_options['first_step']
        _max_step = max_step if max_step is not _unspecified else from_phase.simulate_options['max_step']

        if isinstance(phase_tx, GaussLobatto):
            grid = GaussLobattoGrid(num_segments=num_seg, nodes_per_seg=seg_order, segment_ends=seg_ends,
                                    compressed=compressed)
        elif isinstance(phase_tx, Radau):
            grid = RadauGrid(num_segments=num_seg, nodes_per_seg=seg_order + 1, segment_ends=seg_ends,
                             compressed=compressed)
        elif isinstance(phase_tx.grid_data, GaussLobattoGrid) or \
                isinstance(phase_tx.grid_data, RadauGrid):
            grid = phase_tx.grid_data
        else:
            raise RuntimeError(f'Unexpected grid class for {phase_tx.grid_data}. Only phases with GaussLobatto '
                               f'or Radau grids can be simulated.')

        if times_per_seg is None:
            output_grid = None
        else:
            output_grid = UniformGrid(num_segments=num_seg, nodes_per_seg=times_per_seg, segment_ends=seg_ends,
                                      compressed=compressed)

        tx = ExplicitShooting(propagate_derivs=False,
                              subprob_reports=reports,
                              grid=grid,
                              output_grid=output_grid,
                              method=_method,
                              atol=_atol,
                              rtol=_rtol,
                              first_step=_first_step,
                              max_step=_max_step)

        super().__init__(from_phase=from_phase, transcription=tx, ode_class=from_phase.options['ode_class'],
                         ode_init_kwargs=from_phase.options['ode_init_kwargs'])

        # Remove invalid options
        for state_name, options in self.state_options.items():
            options['fix_final'] = False  # ExplicitShooting will raise if `fix_final` is True for any states.

        # Remove all but the default timeseries object
        self._timeseries = {ts_name: ts_options for ts_name, ts_options in self._timeseries.items()
                            if ts_name == 'timeseries'}

    def set_val_from_phase(self, from_phase):
        """
        Set the necessary values to simulate the phase based on variables in the given phase.

        Parameters
        ----------
        from_phase : Phase
            The dymos phase from which this simulation phase should pull its values.
        """

        t_initial = from_phase.get_val('t_initial', units=self.time_options['units'])
        self.set_val('t_initial', t_initial, units=self.time_options['units'])

        t_duration = from_phase.get_val('t_duration', units=self.time_options['units'])
        self.set_val('t_duration', t_duration, units=self.time_options['units'])

        for name, options in self.state_options.items():
            val = from_phase.get_val(f'states:{name}', units=options['units'])[0, ...]
            self.set_val(f'states:{name}', val, units=options['units'])

        for name, options in self.parameter_options.items():
            val = from_phase.get_val(f'parameters:{name}', units=options['units'])
            self.set_val(f'parameters:{name}', val, units=options['units'])

        for name, options in self.control_options.items():
            val = from_phase.get_val(f'controls:{name}', units=options['units'])
            self.set_val(f'controls:{name}', val, units=options['units'])

        for name, options in self.polynomial_control_options.items():
            val = from_phase.get_val(f'polynomial_controls:{name}', units=options['units'])
            self.set_val(f'polynomial_controls:{name}', val, units=options['units'])

    def initialize_values_from_phase(self, prob, from_phase, phase_path=''):
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
        """
        phs = from_phase

        op_dict = dict([(name, options) for (name, options) in phs.list_outputs(units=True,
                                                                                list_autoivcs=True,
                                                                                out_stream=None)])
        ip_dict = dict([(name, options) for (name, options) in phs.list_inputs(units=True,
                                                                               out_stream=None)])

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
            prob[f'{self_path}states:{name}'][...] = op['val'][0, ...]

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
