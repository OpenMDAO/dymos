from __future__ import division, print_function, absolute_import

from collections import Iterable
from six import iteritems

import numpy as np

from scipy import interpolate

from openmdao.api import Group, IndepVarComp
from openmdao.utils.general_utils import warn_deprecation
from openmdao.core.system import System

from openmdao.utils.logger_utils import get_logger
from dymos.phases.components import BoundaryConstraintComp
from dymos.phases.components import ControlInputComp
from dymos.phases.components import TimeComp
from dymos.phases.components import EndpointConditionsComp
from dymos.phases.options import ControlOptionsDictionary, \
    StateOptionsDictionary, TimeOptionsDictionary
from dymos.phases.components import ControlRateComp
from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos.utils.misc import CoerceDesvar


class PhaseBase(Group):
    def __init__(self, **kwargs):

        super(PhaseBase, self).__init__(**kwargs)

        self.state_options = {}
        self.control_options = {}
        self.time_options = TimeOptionsDictionary()
        self._boundary_constraints = {}
        self._path_constaints = {}
        self._objectives = []
        self._ode_controls = {}
        self.grid_data = None
        self._time_extents = []

        self.ode_options = self.metadata['ode_class'].ode_options

        # Copy default value for options from the ODEOptions
        for state_name, options in iteritems(self.ode_options._states):
            self.state_options[state_name] = StateOptionsDictionary()
            self.state_options[state_name]['shape'] = options['shape']
            self.state_options[state_name]['units'] = options['units']
            self.state_options[state_name]['targets'] = options['targets']
            self.state_options[state_name]['rate_source'] = options['rate_source']

        # Integration variable options default to values from the RHS
        self.time_options['units'] = self.ode_options._time_options['units']
        self.time_options['targets'] = self.ode_options._time_options['targets']

    def initialize(self):
        # Required metadata
        self.metadata.declare('num_segments', types=int, desc='Number of segments')
        self.metadata.declare('ode_class',
                              desc='System defining the ODE')
        self.metadata.declare('ode_init_kwargs', types=dict, default={},
                              desc='Keyword arguments provided when initializing the ODE System')
        self.metadata.declare('transcription', values=['gauss-lobatto', 'radau-ps'],
                              desc='Transcription technique of the optimal control problem.')

        # Optional metadata
        self.metadata.declare(
            'segment_ends', default=None, types=Iterable, allow_none=True,
            desc='Iterable of locations of segment ends or None for equally spaced segments')
        self.metadata.declare(
            'transcription_order', default=3, types=(int, Iterable),
            desc='Order of the transcription')
        self.metadata.declare(
            'compressed', default=False, types=bool, desc='Use compressed transcription')

    def set_state_options(self, name, **kwargs):
        """
        Declares that a control of the RHS is to be used as a dynamic control.

        Parameters
        ----------
        name : str
            Name of the state variable in the RHS.
        units : str or None
            Units in which the state variable is defined.
        opt : bool
            Specifies whether this state is to be a design variable for the optimization.
            If False, then the value is expected to be set by the user or provided externally.
        val :  ndarray
            The default value of the state at the nodes of the phase.
        fix_initial : bool
            If True, omit the first value of the state from the design variables (prevent the
            optimizer from changing it). Default = False.
        fix_final : bool
            If True, omit the final value of the state from the design variables (prevent the
            optimizer from changing it). Default = False.
        lower : float or ndarray
            The lower bound of the state at the nodes of the phase.
        upper : float or ndarray
            The upper bound of the state at the nodes of the phase.
        scaler : float or ndarray
            The scaler of the state value at the nodes of the phase.
        adder : float or ndarray
            The adder of the state value at the nodes of the phase.
        ref0 : float or ndarray
            The zero-reference value of the state at the nodes of the phase.
        ref : float or ndarray
            The unit-reference value of the state at the nodes of the phase
        defect_scaler : float or ndarray
            The scaler of the state defect at the collocation nodes of the phase.
        continuity : bool or dict

        """
        self.state_options[name].update(kwargs)

    def add_control(self, name, val=0.0, units=0, dynamic=True, opt=True, lower=None, upper=None,
                    fix_initial=False, fix_final=False,
                    scaler=None, adder=None, ref=None, ref0=None, continuity=None,
                    rate_continuity=None, rate2_continuity=None,
                    rate_param=None, rate2_param=None):
        """
        Declares that a parameter of the ODE is to potentially be used as an optimal control.

        Parameters
        ----------
        name : str
            Name of the controllable parameter in the ODE.
        val : float or ndarray
            Default value of the control at all nodes.  If val scalar and the control
            is dynamic it will be broadcast.
        units : str or None or 0
            Units in which the control variable is defined.  If 0, use the units declared
            for the parameter in the ODE.
        dynamic : bool
            If True (default) this is a dynamic control, the values provided correspond to
            the number of nodes in the phase.  If False, this is a static control, sized (1,),
            and that value is broadcast to all nodes within the phase.
        opt : bool
            If True (default) the value(s) of this control will be design variables in
            the optimization problem, in the path 'phase_name.indep_controls.controls:control_name'.
            If False, the values of this control will exist in
            'phase_name.input_controls.controls:control_name', where it may be connected to
            external sources if desired.
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
        contiuity : bool or None
            True if continuity in the value of the control is desired at the segment bounds.
            See notes about default values for continuity.
        rate_continuity : bool or None
            True if continuity in the rate of the control is desired at the segment bounds.
            See notes about default values for continuity.
        rate_param : None or str
            The name of the parameter in the ODE to which the first time-derivative
            of the control value is connected.
        rate2_param : None or str
            The name of the parameter in the ODE to which the second time-derivative
            of the control value is connected.

        Notes
        -----
        If continuity is None or rate continuity is None, the default value for
        continuity is True and rate continuity of False.

        The default value of continuity and rate continuity for input controls (opt=False)
        is False.

        The user may override these defaults by specifying them as True or False.

        """
        if name in self.control_options:
            raise ValueError('{0} has already been added as a control.'.format(name))

        self.control_options[name] = ControlOptionsDictionary()

        if name in self.ode_options._dynamic_parameters:
            ode_param_info = self.ode_options._dynamic_parameters[name]
            self.control_options[name]['units'] = ode_param_info['units']
            self.control_options[name]['shape'] = ode_param_info['shape']
        else:
            rate_used = rate_param is not None and \
                rate_param in self.ode_options._dynamic_parameters
            rate2_used = rate2_param is not None and \
                rate2_param in self.ode_options._dynamic_parameters
            if not rate_used and not rate2_used:
                err_msg = 'Control {0} has no valid connection to a controllable parameter ' \
                          'through its value, rate, or second derivative.'.format(name)
                raise ValueError(err_msg)

        if rate_param is not None:
            ode_rate_param_info = self.ode_options._dynamic_parameters[rate_param]
            self.control_options[name]['rate_param'] = rate_param
            self.control_options[name]['shape'] = ode_rate_param_info['shape']
        if rate2_param is not None:
            ode_rate2_param_info = self.ode_options._dynamic_parameters[rate2_param]
            self.control_options[name]['rate2_param'] = rate2_param
            self.control_options[name]['shape'] = ode_rate2_param_info['shape']

        # Don't allow the user to provide desvar options if the control is not optimal
        if not opt:
            illegal_options = []
            if lower is not None:
                illegal_options.append('lower')
            if upper is not None:
                illegal_options.append('upper')
            if scaler is not None:
                illegal_options.append('scaler')
            if adder is not None:
                illegal_options.append('adder')
            if ref is not None:
                illegal_options.append('ref')
            if ref0 is not None:
                illegal_options.append('ref0')
            if continuity is not None:
                illegal_options.append('continuity')
            if rate_continuity is not None:
                illegal_options.append('rate_continuity')
            if illegal_options:
                msg = 'Invalid options for non-optimal control:' + ', '.join(illegal_options)
                raise ValueError(msg)

        self.control_options[name]['val'] = val
        self.control_options[name]['dynamic'] = dynamic
        self.control_options[name]['opt'] = opt
        # self.control_options[name]['opt_initial'] = opt_initial
        # self.control_options[name]['opt_final'] = opt_final
        self.control_options[name]['fix_initial'] = fix_initial
        self.control_options[name]['fix_final'] = fix_final
        self.control_options[name]['lower'] = lower
        self.control_options[name]['upper'] = upper
        self.control_options[name]['scaler'] = scaler
        self.control_options[name]['adder'] = adder
        self.control_options[name]['ref'] = ref
        self.control_options[name]['ref0'] = ref0

        if continuity is None:
            self.control_options[name]['continuity'] = opt
        else:
            self.control_options[name]['continuity'] = continuity

        if rate_continuity is None:
            self.control_options[name]['rate_continuity'] = False
        else:
            self.control_options[name]['rate_continuity'] = rate_continuity

        if rate2_continuity is None:
            self.control_options[name]['rate2_continuity'] = False
        else:
            self.control_options[name]['rate2_continuity'] = rate2_continuity

        if units != 0:
            self.control_options[name]['units'] = units

    def add_boundary_constraint(self, name, loc, constraint_name=None, **kwargs):
        r"""
        Add a constraint variable to this system.

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
        **kwargs : optional
            Keyword arguments that are saved as metadata for the
            design variable.
        """
        if 'indices' in kwargs:
            raise ValueError('boundary constraint does not accept indices')

        if loc not in ['initial', 'final']:
            raise ValueError('invalid boundary constraint location {0}'.format(loc))

        if constraint_name is None:
            constraint_name = name.split('.')[-1]

        if name not in self._boundary_constraints:
            self._boundary_constraints[name] = {}
            self._boundary_constraints[name]['constraint_name'] = constraint_name

        if loc not in self._boundary_constraints[name]:
            self._boundary_constraints[name][loc] = {}

        self._boundary_constraints[name][loc].update(kwargs)

    def add_path_constraint(self, name, constraint_name=None, **kwargs):
        r"""
        Add a constraint variable to this system.

        Parameters
        ----------
        name : string
            Name of the response variable in the system.
        constraint_name : string or None
            The name of the variable as provided to the boundary constraint comp.  By
            default this is the last element in `name` when split by dots.  The user may
            override the constraint name if splitting the path causes name collisions.
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
        **kwargs : optional
            Keyword arguments that are saved as metadata for the
            design variable.
        """
        if constraint_name is None:
            constraint_name = name.split('.')[-1]

        if name not in self._path_constaints:
            self._path_constaints[name] = {}
            self._path_constaints[name]['constraint_name'] = constraint_name

        self._path_constaints[name].update(kwargs)

    def set_objective(self, name, loc='final', index=None, shape=(1,), ref=None, ref0=None,
                      adder=None, scaler=None, parallel_deriv_color=None,
                      vectorize_derivs=False, simul_coloring=None, simul_map=None):
        """
        Allows the user to set an objective in the phase.  If name is not a state,
        control, or 'time', then this is assumed to be the path of the variable
        to be constrained in the RHS.

        The default OpenMDAO `add_objective` method may still be used with the correct
        path name to the response, but this method is intended to be
        transcription-independent.

        Parameters
        ----------
        name : str
            Name of the response variable in the system.
        loc : str
            Where in the phase the objective is to be evaluated.  Valid
            options are 'start' and 'end'.  The default is 'end'.
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        index : int, optional
            If variable is an array, this indicates which entry is of
            interest for this particular response. This may be a positive
            or negative integer.  If present, this overrides loc.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.

        """
        warn_deprecation('set_objective has been replaced with add_objective')
        self.add_objective(name, loc=loc, index=index, shape=shape, ref=ref, ref0=ref0,
                           adder=adder, scaler=scaler, parallel_deriv_color=parallel_deriv_color,
                           vectorize_derivs=vectorize_derivs, simul_coloring=simul_coloring,
                           simul_map=simul_map)

    def _add_objective(self, obj_path, loc='final', index=None, shape=(1,), ref=None, ref0=None,
                       adder=None, scaler=None, parallel_deriv_color=None,
                       vectorize_derivs=False, simul_coloring=None, simul_map=None):
        """
        Called by add_objective in classes that derive from PhaseBase.  Each subclass is responsible
        for determining the objective paht in the system.  This method then figures out the correct
        index based on the given loc and index attributes, and calls the standard add_objective
        method.

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


        Parameters
        ----------
        obj_path : str
            The name of the variable in the phase to be used as an objective.
        loc : str
            One of 'initial' or 'final', depending on where in the phase the objective should be
            measured.
        index : int or None
            The index into the flattened shape giving the index at an instance in time to be used
            as the objective.  This index assumes row-major (C) ordering when flattening.
        shape : tuple
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
        simul_coloring : ndarray or list of int
            An array or list of integer color values.  Must match the size of the
            objective variable.
        simul_map : dict
            Mapping of this response to each design variable where simultaneous derivs will
            be used.  Each design variable entry is another dict keyed on color, and the values
            in the color dict are tuples of the form (resp_idxs, color_idxs).
        """
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
                             'one of \'initial\' or \'final\'.')

        super(PhaseBase, self).add_objective(obj_path, ref=ref, ref0=ref0, index=obj_index,
                                             adder=adder, scaler=scaler,
                                             parallel_deriv_color=parallel_deriv_color,
                                             vectorize_derivs=vectorize_derivs,
                                             simul_coloring=simul_coloring,
                                             simul_map=simul_map)

    def set_time_options(self, **kwargs):
        """
        Set options for the time (or the integration variable) in the Phase.

        Parameters
        ----------
        opt_initial : bool
            If True, the initial time of the phase is a design variable
            for optimization, otherwise False.
        opt_initial : bool
            If True, the duration of the phase is a design variable
            for optimization, otherwise False.
        initial : float
            Value of the time at the start of the phase.
        initial_bounds : Iterable of size 2
            Tuple of (lower, upper) bounds for time at the start of the phase.
        initial_scaler : float
            Scalar for the initial value of time.
        initial_adder : float
            Adder for the initial value of time.
        inital_ref0 : float
            Zero-reference value for the initial value of time.
        initial_ref : float
            Unit-reference value for the initial value of time.
        duration : float
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
        # Don't allow the user to provide desvar options if the control is not optimal
        if not kwargs.get('opt_initial', True):
            illegal_options = []
            if kwargs.get('initial_bounds', None) is not None:
                illegal_options.append('initial_bounds')
            if kwargs.get('initial_scaler', None) is not None:
                illegal_options.append('initial_scaler')
            if kwargs.get('initial_adder', None) is not None:
                illegal_options.append('initial_adder')
            if kwargs.get('initial_ref', None) is not None:
                illegal_options.append('initial_ref')
            if kwargs.get('initial_ref0', None) is not None:
                illegal_options.append('initial_ref0')
            if illegal_options:
                msg = 'Invalid options for non-optimal ' \
                      'initial time: {0}'.format(', '.join(illegal_options))
                raise ValueError(msg)
        if not kwargs.get('opt_duration', True):
            illegal_options = []
            if kwargs.get('duration_bounds', None) is not None:
                illegal_options.append('duration_bounds')
            if kwargs.get('duration_scaler', None) is not None:
                illegal_options.append('duration_scaler')
            if kwargs.get('duration_adder', None) is not None:
                illegal_options.append('duration_adder')
            if kwargs.get('duration_ref', None) is not None:
                illegal_options.append('duration_ref')
            if kwargs.get('duration_ref0', None) is not None:
                illegal_options.append('duration_ref0')
            if illegal_options:
                msg = 'Invalid options for non-optimal ' \
                      'duration: {0}'.format(', '.join(illegal_options))
                raise ValueError(msg)
        self.time_options.update(kwargs)

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
            'time', 'state', 'control', 'control_rate',
            'control_rate2', or 'rhs'.

        """
        if var == 'time':
            return 'time'
        elif var in self.state_options:
            return 'state'
        elif var in self.control_options:
            if self.control_options[var]['opt']:
                return 'indep_control'
            else:
                return 'input_control'
        elif var.endswith('_rate'):
            if var[:-5] in self.control_options:
                return 'control_rate'
        elif var.endswith('_rate2'):
            if var[:-6] in self.control_options:
                return 'control_rate2'
        else:
            return 'rhs'

    def setup(self):
        transcription = self.metadata['transcription']
        num_segments = self.metadata['num_segments']
        transcription_order = self.metadata['transcription_order']
        segment_ends = self.metadata['segment_ends']
        compressed = self.metadata['compressed']

        if np.any(np.asarray(transcription_order) < 3):
            raise ValueError('Given transcription order ({0}) is less than '
                             'the minimum allowed value (3)'.format(transcription_order))

        self.grid_data = grid_data = GridData(
            num_segments=num_segments, transcription=transcription,
            transcription_order=transcription_order,
            segment_ends=segment_ends,
            compressed=compressed)

        self._time_extents = self._setup_time()

        # Declare control_rate comp to which we'll connect controls and parameters.
        if self.control_options:
            ctrl_rate_comp = ControlRateComp(control_options=self.control_options,
                                             time_units=self.time_options['units'],
                                             grid_data=self.grid_data)

            promoted_outputs = []

            self._setup_controls()
            # self._setup_opt_controls()
            # self._setup_input_controls()
            promoted_outputs.append('control_rates:*')

            self.add_subsystem('control_rate_comp', subsys=ctrl_rate_comp,
                               promotes_outputs=promoted_outputs)
            self.connect('time.dt_dstau', 'control_rate_comp.dt_dstau')

        self._setup_rhs()
        self._setup_defects()
        self._setup_states()

        self._setup_endpoint_conditions()
        self._setup_boundary_constraints()
        self._setup_path_constraints()

        self._check_unprovided_controls()

    def _setup_time(self):
        """
        Setup up the time component and time extents for the phase.

        Returns
        -------
        comps
            A list of the component names needed for time extents.
        """
        time_units = self.time_options['units']
        grid_data = self.grid_data

        indeps = []
        externals = []
        comps = []

        if self.time_options['opt_initial']:
            indeps.append('t_initial')
            self.connect('t_initial', 'time.t_initial')
        else:
            externals.append('t_initial')

        if self.time_options['opt_duration']:
            indeps.append('t_duration')
            self.connect('t_duration', 'time.t_duration')
        else:
            externals.append('t_duration')

        if indeps:
            indep = IndepVarComp()
            for var in indeps:
                indep.add_output(var, val=1.0, units=time_units)
            self.add_subsystem('time_extents', indep, promotes_outputs=['*'])
            comps += ['time_extents']

        time_comp = TimeComp(grid_data=grid_data, units=time_units)
        self.add_subsystem('time', time_comp, promotes_outputs=['time'], promotes_inputs=externals)

        if self.time_options['opt_initial']:
            self.add_design_var('t_initial',
                                lower=self.time_options['initial_bounds'][0],
                                upper=self.time_options['initial_bounds'][1],
                                scaler=self.time_options['initial_scaler'],
                                adder=self.time_options['initial_adder'],
                                ref0=self.time_options['initial_ref0'],
                                ref=self.time_options['initial_ref'])

        if self.time_options['opt_duration']:
            self.add_design_var('t_duration',
                                lower=self.time_options['duration_bounds'][0],
                                upper=self.time_options['duration_bounds'][1],
                                scaler=self.time_options['duration_scaler'],
                                adder=self.time_options['duration_adder'],
                                ref0=self.time_options['duration_ref0'],
                                ref=self.time_options['duration_ref'])
        return comps

    def _setup_controls(self):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        opt_controls = [name for (name, opts) in iteritems(self.control_options) if opts['opt']]

        num_opt_controls = len(opt_controls)

        num_input_controls = len(self.control_options) - num_opt_controls

        grid_data = self.grid_data

        if num_opt_controls > 0:
            indep = self.add_subsystem('indep_controls', subsys=IndepVarComp(),
                                       promotes_outputs=['*'])

        if num_input_controls > 0:
            passthru = ControlInputComp(num_nodes=grid_data.num_nodes,
                                        control_options=self.control_options)

            self.add_subsystem('input_controls', subsys=passthru, promotes_inputs=['*'],
                               promotes_outputs=['*'])

        num_dynamic_controls = 0

        for name, options in iteritems(self.control_options):
            if options['opt']:
                if options['dynamic']:
                    num_dynamic_controls = num_dynamic_controls + 1
                    num_input_nodes = grid_data.num_dynamic_control_input_nodes
                    map_indices_to_all = self.grid_data.input_maps['dynamic_control_to_all']

                    desvar_indices = list(range(self.grid_data.num_dynamic_control_input_nodes))
                    if options['fix_initial']:
                        desvar_indices.pop(0)
                    if options['fix_final']:
                        desvar_indices.pop()

                    if len(desvar_indices) > 0:
                        coerce_desvar_option = CoerceDesvar(grid_data.subset_num_nodes['disc'],
                                                            desvar_indices, options)

                        self.add_design_var(name='controls:{0}'.format(name),
                                            lower=coerce_desvar_option('lower'),
                                            upper=coerce_desvar_option('upper'),
                                            scaler=coerce_desvar_option('scaler'),
                                            adder=coerce_desvar_option('adder'),
                                            ref0=coerce_desvar_option('ref0'),
                                            ref=coerce_desvar_option('ref'),
                                            indices=desvar_indices)
                # END DYNAMIC CONTROL
                else:
                    # Static control
                    num_input_nodes = 1
                    map_indices_to_all = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)

                    self.add_design_var(name='controls:{0}'.format(name),
                                        lower=options['lower'],
                                        upper=options['upper'],
                                        scaler=options['scaler'],
                                        adder=options['adder'],
                                        ref0=options['ref0'],
                                        ref=options['ref'])

                indep.add_output(name='controls:{0}'.format(name),
                                 val=options['val'],
                                 shape=(num_input_nodes, np.prod(options['shape'])),
                                 units=options['units'])
                # END STATIC CONTROL
                control_src_name = 'controls:{0}'.format(name)

            # END OPTIMAL CONTROL
            else:
                if options['dynamic']:
                    map_indices_to_all = self.grid_data.input_maps['dynamic_control_to_all']
                else:
                    map_indices_to_all = np.zeros(self.grid_data.subset_num_nodes['all'],
                                                  dtype=int)
                control_src_name = 'controls:{0}_out'.format(name)

            # END INPUT CONTROL

            # Connect to control rate
            if options['dynamic']:
                self.connect(control_src_name,
                             'control_rate_comp.controls:{0}'.format(name),
                             src_indices=map_indices_to_all)

        return num_dynamic_controls

    def _setup_rhs(self):
        raise NotImplementedError()

    def _setup_defects(self):
        raise NotImplementedError()

    def _setup_states(self):
        raise NotImplementedError()

    def _setup_endpoint_conditions(self):

        jump_comp = self.add_subsystem('indep_jumps', subsys=IndepVarComp(),
                                       promotes_outputs=['*'])

        jump_comp.add_output('initial_jump:time', val=0.0, units=self.time_options['units'],
                             desc='discontinuity in time at the start of the phase')

        jump_comp.add_output('final_jump:time', val=0.0, units=self.time_options['units'],
                             desc='discontinuity in time at the end of the phase')

        endpoint_comp = EndpointConditionsComp(time_options=self.time_options,
                                               state_options=self.state_options,
                                               control_options=self.control_options)

        self.connect('time',
                     'endpoint_conditions.initial_value:time',
                     src_indices=np.array([0]))

        self.connect('time',
                     'endpoint_conditions.final_value:time',
                     src_indices=np.array([-1]))

        self.connect('initial_jump:time',
                     'endpoint_conditions.initial_jump:time')

        self.connect('final_jump:time',
                     'endpoint_conditions.final_jump:time')

        promoted_list = ['time--', 'time-+', 'time+-', 'time++']

        for state_name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])
            ar = np.arange(size)

            jump_comp.add_output('initial_jump:{0}'.format(state_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'start of the phase'.format(state_name))

            jump_comp.add_output('final_jump:{0}'.format(state_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'end of the phase'.format(state_name))

            self.connect('states:{0}'.format(state_name),
                         'endpoint_conditions.initial_value:{0}'.format(state_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('states:{0}'.format(state_name),
                         'endpoint_conditions.final_value:{0}'.format(state_name),
                         src_indices=-ar[::-1]-1, flat_src_indices=True)

            self.connect('initial_jump:{0}'.format(state_name),
                         'endpoint_conditions.initial_jump:{0}'.format(state_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('final_jump:{0}'.format(state_name),
                         'endpoint_conditions.final_jump:{0}'.format(state_name),
                         src_indices=ar, flat_src_indices=True)

            promoted_list += ['states:{0}--'.format(state_name),
                              'states:{0}-+'.format(state_name),
                              'states:{0}+-'.format(state_name),
                              'states:{0}++'.format(state_name)]

        for control_name, options in iteritems(self.control_options):
            size = np.prod(options['shape'])
            ar = np.arange(size)

            if options['opt']:
                suffix = ''
            else:
                suffix = '_out'

            jump_comp.add_output('initial_jump:{0}'.format(control_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'start of the phase'.format(control_name))

            jump_comp.add_output('final_jump:{0}'.format(control_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'end of the phase'.format(control_name))

            self.connect('controls:{0}{1}'.format(control_name, suffix),
                         'endpoint_conditions.initial_value:{0}'.format(control_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('controls:{0}{1}'.format(control_name, suffix),
                         'endpoint_conditions.final_value:{0}'.format(control_name),
                         src_indices=-ar[::-1]-1, flat_src_indices=True)

            self.connect('initial_jump:{0}'.format(control_name),
                         'endpoint_conditions.initial_jump:{0}'.format(control_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('final_jump:{0}'.format(control_name),
                         'endpoint_conditions.final_jump:{0}'.format(control_name),
                         src_indices=ar, flat_src_indices=True)

            promoted_list += ['controls:{0}--'.format(control_name),
                              'controls:{0}-+'.format(control_name),
                              'controls:{0}+-'.format(control_name),
                              'controls:{0}++'.format(control_name)]

        self.add_subsystem(name='endpoint_conditions', subsys=endpoint_comp,
                           promotes_outputs=promoted_list)

    def _setup_boundary_constraints(self):
        """
        Adds BoundaryConstraintComp if necessary and issues appropriate connections.
        """
        transcription = self.metadata['transcription']
        bc_comp = None

        if self._boundary_constraints:
            bc_comp = self.add_subsystem('boundary_constraints', subsys=BoundaryConstraintComp())

        for var, options in iteritems(self._boundary_constraints):
            con_name = options['constraint_name']
            con_units = options.get('units', None)
            con_shape = options.get('shape', (1,))

            # Determine the path to the variable which we will be constraining
            var_type = self._classify_var(var)

            if var_type == 'time':
                options['shape'] = (1,)
                options['units'] = self.time_options['units'] if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'time'
            elif var_type == 'state':
                state_shape = self.state_options[var]['shape']
                state_units = self.state_options[var]['units']
                options['shape'] = state_shape if con_shape is None else con_shape
                options['units'] = state_units if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'states:{0}'.format(var)
            elif var_type == 'indep_control':
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape if con_shape is None else con_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'controls:{0}'.format(var)
            elif var_type == 'input_control':
                control_shape = self.control_options[var]['shape']
                control_units = self.control_options[var]['units']
                options['shape'] = control_shape if con_shape is None else con_shape
                options['units'] = control_units if con_units is None else con_units
                options['linear'] = True
                constraint_path = 'controls:{0}_out'.format(var)
            elif var_type == 'control_rate':
                control_var = var[:-5]
                control_shape = self.control_options[control_var]['shape']
                control_units = self.control_options[control_var]['units']
                control_rate_units = get_rate_units(control_units,
                                                    self.time_options['units'],
                                                    deriv=1)
                options['shape'] = control_shape if con_shape is None else con_shape
                options['units'] = control_rate_units if con_units is None else con_units
                constraint_path = 'control_rates:{0}'.format(var)
            elif var_type == 'control_rate2':
                control_var = var[:-6]
                control_shape = self.control_options[control_var]['shape']
                control_units = self.control_options[control_var]['units']
                control_rate_units = get_rate_units(control_units,
                                                    self.time_options['units'],
                                                    deriv=2)
                options['shape'] = control_shape if con_shape is None else con_shape
                options['units'] = control_rate_units if con_units is None else con_units
                constraint_path = 'control_rates:{0}'.format(var)
            else:
                # Failed to find variable, assume it is in the RHS
                if transcription == 'gauss-lobatto':
                    constraint_path = 'rhs_disc.{0}'.format(var)
                elif transcription == 'radau-ps':
                    constraint_path = 'rhs_all.{0}'.format(var)
                else:
                    raise ValueError('Invalid transcription')

                options['shape'] = con_shape
                options['units'] = con_units

            if 'initial' in options:
                options['initial']['units'] = options['units']
                bc_comp._add_initial_constraint(con_name,
                                                **options['initial'])
            if 'final' in options:
                options['final']['units'] = options['units']
                bc_comp._add_final_constraint(con_name,
                                              **options['final'])

            # Build the correct src_indices regardless of shape
            size = np.prod(options['shape'])
            src_idxs_initial = np.arange(size, dtype=int).reshape(options['shape'])
            src_idxs_final = np.arange(-size, 0, dtype=int).reshape(options['shape'])
            src_idxs = np.stack((src_idxs_initial, src_idxs_final))

            self.connect(constraint_path,
                         'boundary_constraints.boundary_values:{0}'.format(con_name),
                         src_indices=src_idxs, flat_src_indices=True)

    def _check_unprovided_controls(self):
        logger = get_logger('check_config', use_format=True)
        unconnected = []
        ode_options = self.metadata['ode_class'](num_nodes=1,
                                                 **self.metadata['ode_init_kwargs']).ode_options
        ode_parameters = ode_options._dynamic_parameters.copy()
        ode_parameters.update(ode_options._static_parameters)

        for p in ode_parameters:
            p_is_connected = False
            for control_name, options in iteritems(self.control_options):
                if control_name == p:
                    p_is_connected = True
                if options['rate_param'] == p:
                    p_is_connected = True
                if options['rate2_param'] == p:
                    p_is_connected = True
            if not p_is_connected:
                unconnected.append(p)
        if unconnected:
            logger.warning('The following ODE parameters are not provided'
                           ' by phase {0} as controls, control rates nor'
                           ' parameters: {1}'.format(self.name, unconnected))

    def get_values(self, var, nodes='all'):
        """
        Retrieve the values of the given variable at the given
        subset of nodes.

        Parameters
        ----------
        var : str
            The variable whose values are to be returned.  This may be
            the name 'time', the name of a state, control, or parameter,
            or the path to a variable in the ODE system of the phase.
        nodes : str
            The name of a node subset, one of 'disc', 'col', or 'all'.
            The default is 'all'.

        Returns
        -------
        ndarray
            An array of the values at the requested node subset.  The
            node index is the first dimension of the ndarray.
        """
        raise NotImplementedError('get_values has not been implemented for this class.')

    def interpolate(self, xs=None, ys=None, nodes=None, kind='linear'):
        """
        Return an array of values on [a,b] linearly interpolated to the
        input nodes of the phase.

        Parameters
        ----------
        xs :  ndarray
            Array of integration variable values.
        ys :  ndarray
            Array of control/state/parameter values.
        nodes :  str
            Node type ('disc', 'col' or 'all').
        kind : str
            Specifies the kind of interpolation, as per the scipy.interpolate package.
            One of ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of zeroth, first, second or third order) or as an
            integer specifying the order of the spline interpolator to use.
            Default is 'linear'.

        Returns
        -------
        np.array
            The values of y interpolated at nodes of the specified type.
        """
        if not isinstance(ys, Iterable):
            raise ValueError('ys must be provided as an Iterable of length at least 2.')
        if nodes not in ('col', 'disc', 'all'):
            raise ValueError("nodes must be one of 'col', 'disc', or 'all'")
        if xs is None:
            if len(ys) != 2:
                raise ValueError('xs may only be unspecified when len(ys)=2')
            if kind != 'linear':
                raise ValueError('kind must be linear when xs is unspecified.')
            xs = [-1, 1]
        node_locations = self.grid_data.node_ptau[self.grid_data.subset_node_indices[nodes]]
        if self.metadata['compressed']:
            node_locations = np.array(sorted(list(set(node_locations))))
        # Affine transform xs into tau space [-1, 1]
        _xs = np.asarray(xs)
        m = 2.0/(_xs[-1] - _xs[0])
        b = 1.0-(m*_xs[-1])
        taus = m*_xs + b
        interpfunc = interpolate.interp1d(taus, ys, kind=kind)
        return np.atleast_2d(interpfunc(node_locations)).T
