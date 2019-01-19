from __future__ import division, print_function, absolute_import

from collections import Iterable
import inspect
from six import iteritems
import warnings

import numpy as np

from scipy import interpolate

from openmdao.api import Group, IndepVarComp
from openmdao.utils.general_utils import warn_deprecation
from openmdao.core.system import System

# from dymos.phases.components import BoundaryConstraintComp
from dymos.phases.components import BoundaryConstraintComp
from dymos.phases.components import InputParameterComp
from dymos.phases.components import TimeComp
from dymos.phases.options import ControlOptionsDictionary, DesignParameterOptionsDictionary, \
    InputParameterOptionsDictionary, StateOptionsDictionary, TimeOptionsDictionary
from dymos.phases.components import ControlInterpComp
from dymos.phases.grid_data import GridData
from dymos.ode_options import ODEOptions
from dymos.utils.constants import INF_BOUND
from dymos.utils.misc import get_rate_units
from dymos.utils.misc import CoerceDesvar


_unspecified = object()


class PhaseBase(Group):
    def __init__(self, **kwargs):

        super(PhaseBase, self).__init__(**kwargs)

        self.state_options = {}
        self.control_options = {}
        self.design_parameter_options = {}
        self.input_parameter_options = {}
        self.time_options = TimeOptionsDictionary()
        self._initial_boundary_constraints = {}
        self._final_boundary_constraints = {}
        self._path_constraints = {}
        self._objectives = []
        self._ode_controls = {}
        self.grid_data = None
        self._time_extents = []

        # check that ode_class is appropriate
        if not inspect.isclass(self.options['ode_class']):
            raise ValueError('ode_class must be a class, not an instance.')
        if not issubclass(self.options['ode_class'], System):
            raise ValueError('ode_class must be derived from openmdao.core.System.')
        if not hasattr(self.options['ode_class'], 'ode_options') or \
                not isinstance(self.options['ode_class'].ode_options, ODEOptions):
            raise ValueError('ode_class has no ODE metadata.  Use @declare_time, @declare_state'
                             'and @declare_control to assign ODE metadata.')

        self.ode_options = self.options['ode_class'].ode_options

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
        self.time_options['time_phase_targets'] = \
            self.ode_options._time_options['time_phase_targets']
        self.time_options['t_initial_targets'] = \
            self.ode_options._time_options['t_initial_targets']
        self.time_options['t_duration_targets'] = \
            self.ode_options._time_options['t_duration_targets']

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

    def set_state_options(self, name, units=_unspecified, val=1.0,
                          fix_initial=False, fix_final=False, initial_bounds=None,
                          final_bounds=None, lower=None, upper=None, scaler=None, adder=None,
                          ref=None, ref0=None, defect_scaler=1.0, defect_ref=None):
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

        """
        if units is not _unspecified:
            self.state_options[name]['units'] = units
        self.state_options[name]['val'] = val
        self.state_options[name]['fix_initial'] = fix_initial
        self.state_options[name]['fix_final'] = fix_final
        self.state_options[name]['initial_bounds'] = initial_bounds
        self.state_options[name]['final_bounds'] = final_bounds
        self.state_options[name]['lower'] = lower
        self.state_options[name]['upper'] = upper
        self.state_options[name]['scaler'] = scaler
        self.state_options[name]['adder'] = adder
        self.state_options[name]['ref'] = ref
        self.state_options[name]['ref0'] = ref0
        self.state_options[name]['defect_scaler'] = defect_scaler
        self.state_options[name]['defect_ref'] = defect_ref

    def add_control(self, name, val=0.0, units=0, opt=True, lower=None, upper=None,
                    fix_initial=False, fix_final=False,
                    scaler=None, adder=None, ref=None, ref0=None, continuity=None,
                    rate_continuity=None, rate_continuity_scaler=1.0,
                    rate2_continuity=None, rate2_continuity_scaler=1.0,
                    rate_param=None, rate2_param=None):
        """
        Adds a dynamic control variable to be tied to a parameter in the ODE.

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
        continuity : bool or None
            True if continuity in the value of the control is desired at the segment bounds.
            See notes about default values for continuity.
        rate_continuity : bool or None
            True if continuity in the rate of the control is desired at the segment bounds.  This
            rate is normalized to segment tau space.
            See notes about default values for continuity.
        rate_continuity_scaler : float or ndarray
            The scaler to use for the rate_continuity constraint given to the optimizer.
        rate2_continuity : bool or None
            True if continuity in the second derivative of the control is desired at the
            segment bounds. This second derivative is normalized to segment tau space.
            See notes about default values for continuity.
        rate2_continuity_scaler : float or ndarray
            The scaler to use for the rate2_continuity constraint given to the optimizer.
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
        ode_params = self.ode_options._parameters
        if name in self.control_options:
            raise ValueError('{0} has already been added as a control.'.format(name))
        if name in self.design_parameter_options:
            raise ValueError('{0} has already been added as a design parameter.'.format(name))
        if name in self.input_parameter_options:
            raise ValueError('{0} has already been added as an input parameter.'.format(name))
        if name in ode_params and not ode_params[name]['dynamic']:
            raise ValueError('{0} is declared as a static parameter and therefore cannot be '
                             'used as a dynamic control'.format(name))

        self.control_options[name] = ControlOptionsDictionary()

        if name in ode_params:
            ode_param_info = ode_params[name]
            self.control_options[name]['units'] = ode_param_info['units']
            self.control_options[name]['shape'] = ode_param_info['shape']
        else:
            rate_used = \
                rate_param is not None and rate_param in ode_params
            rate2_used = \
                rate2_param is not None and rate2_param in ode_params
            if not rate_used and not rate2_used:
                err_msg = '{0} is not a controllable parameter in the ODE system, nor is it ' \
                          'connected to one through its rate or second derivative.'.format(name)
                raise ValueError(err_msg)

        if rate_param is not None:
            ode_rate_param_info = ode_params[rate_param]
            self.control_options[name]['rate_param'] = rate_param
            self.control_options[name]['shape'] = ode_rate_param_info['shape']
        if rate2_param is not None:
            ode_rate2_param_info = ode_params[rate2_param]
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
                msg = 'Invalid options for non-optimal control "{0}":'.format(name) + \
                      ', '.join(illegal_options)
                warnings.warn(msg, RuntimeWarning)

        self.control_options[name]['val'] = val
        self.control_options[name]['opt'] = opt
        self.control_options[name]['fix_initial'] = fix_initial
        self.control_options[name]['fix_final'] = fix_final
        self.control_options[name]['lower'] = lower
        self.control_options[name]['upper'] = upper
        self.control_options[name]['scaler'] = scaler
        self.control_options[name]['adder'] = adder
        self.control_options[name]['ref'] = ref
        self.control_options[name]['ref0'] = ref0
        self.control_options[name]['rate_continuity_scaler'] = rate_continuity_scaler
        self.control_options[name]['rate2_continuity_scaler'] = rate2_continuity_scaler

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

    def add_design_parameter(self, name, val=0.0, units=0, opt=True,
                             lower=None, upper=None, scaler=None, adder=None, ref=None, ref0=None):
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

        """
        if name in self.control_options:
            raise ValueError('{0} has already been added as a control.'.format(name))
        if name in self.design_parameter_options:
            raise ValueError('{0} has already been added as a design parameter.'.format(name))
        if name in self.input_parameter_options:
            raise ValueError('{0} has already been added as an input parameter.'.format(name))

        self.design_parameter_options[name] = DesignParameterOptionsDictionary()

        if name in self.ode_options._parameters:
            ode_param_info = self.ode_options._parameters[name]
            self.design_parameter_options[name]['units'] = ode_param_info['units']
            self.design_parameter_options[name]['shape'] = ode_param_info['shape']
            self.design_parameter_options[name]['dynamic'] = ode_param_info['dynamic']
        else:
            err_msg = '{0} is not a controllable parameter in the ODE system.'.format(name)
            raise ValueError(err_msg)

        # Don't allow the user to provide desvar options if the design parameter is not a desvar
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
            if illegal_options:
                msg = 'Invalid options for non-optimal design parameter "{0}":'.format(name) \
                      + ', '.join(illegal_options)
                warnings.warn(msg, RuntimeWarning)

        self.design_parameter_options[name]['val'] = val
        self.design_parameter_options[name]['opt'] = opt
        self.design_parameter_options[name]['lower'] = lower
        self.design_parameter_options[name]['upper'] = upper
        self.design_parameter_options[name]['scaler'] = scaler
        self.design_parameter_options[name]['adder'] = adder
        self.design_parameter_options[name]['ref'] = ref
        self.design_parameter_options[name]['ref0'] = ref0

        if units != 0:
            self.design_parameter_options[name]['units'] = units

    def add_input_parameter(self, name, val=0.0, units=0, alias=None):
        """
        Add a design parameter (static control variable) to the phase.

        Parameters
        ----------
        name : str
            Name of the ODE parameter to be controlled via this input parameter.
        val : float or ndarray
            Default value of the design parameter at all nodes.
        units : str or None or 0
            Units in which the design parameter is defined.  If 0, use the units declared
            for the parameter in the ODE.
        alias : str or None
            If provided, specifies the alternative name by which this input parameter is to be
            referenced.

        """

        if name in self.control_options:
            raise ValueError('{0} has already been added as a control.'.format(name))
        if name in self.design_parameter_options:
            raise ValueError('{0} has already been added as a design parameter.'.format(name))
        if name in self.input_parameter_options:
            raise ValueError('{0} has already been added as an input parameter.'.format(name))
        if alias in self.control_options:
            raise ValueError('{0} has already been added as a control.'.format(name))
        if alias in self.design_parameter_options:
            raise ValueError('{0} has already been added as a design parameter.'.format(name))
        if alias in self.input_parameter_options:
            raise ValueError('{0} has already been added as an input parameter.'.format(name))

        ip_name = name if alias is None else alias

        self.input_parameter_options[ip_name] = InputParameterOptionsDictionary()

        if name in self.ode_options._parameters:
            ode_param_info = self.ode_options._parameters[name]
            self.input_parameter_options[ip_name]['units'] = ode_param_info['units']
            self.input_parameter_options[ip_name]['shape'] = ode_param_info['shape']
            self.input_parameter_options[ip_name]['dynamic'] = ode_param_info['dynamic']
        else:
            err_msg = '{0} is not a controllable parameter in the ODE system.'.format(name)
            raise ValueError(err_msg)

        self.input_parameter_options[ip_name]['val'] = val
        self.input_parameter_options[ip_name]['target_param'] = name

        if units != 0:
            self.input_parameter_options[ip_name]['units'] = units

    def add_boundary_constraint(self, name, loc, constraint_name=None, units=None, lower=None,
                                upper=None, equals=None, scaler=None, adder=None,
                                ref=None, ref0=None, linear=False):
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

        bc_dict[name]['lower'] = lower
        bc_dict[name]['upper'] = upper
        bc_dict[name]['equals'] = equals
        bc_dict[name]['scaler'] = scaler
        bc_dict[name]['adder'] = adder
        bc_dict[name]['ref0'] = ref0
        bc_dict[name]['ref'] = ref
        bc_dict[name]['linear'] = linear
        bc_dict[name]['units'] = units

    def add_path_constraint(self, name, constraint_name=None, units=None, lower=None,
                            upper=None, equals=None, scaler=None, adder=None,
                            ref=None, ref0=None, linear=False):
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
        self._path_constraints[name]['linear'] = linear
        self._path_constraints[name]['units'] = units

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
        raise NotImplementedError('This class does not implement add_objective')

    def _add_objective(self, obj_path, loc='final', index=None, shape=(1,), ref=None, ref0=None,
                       adder=None, scaler=None, parallel_deriv_color=None, vectorize_derivs=False):
        """
        Called by add_objective in classes that derive from PhaseBase.  Each subclass is responsible
        for determining the objective path in the system.  This method then figures out the correct
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
                                             vectorize_derivs=vectorize_derivs)

    def set_time_options(self, opt_initial=None, opt_duration=None, fix_initial=False,
                         fix_duration=False, input_initial=False, input_duration=False,
                         initial=0.0, initial_bounds=(None, None), initial_scaler=None,
                         initial_adder=None, initial_ref=None, initial_ref0=None,
                         duration=1.0, duration_bounds=(None, None),
                         duration_scaler=None, duration_adder=None, duration_ref=None,
                         duration_ref0=None):
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
        initial : float
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
        if opt_initial is not None:
            self.time_options['fix_initial'] = not opt_initial
            warn_deprecation('opt_initial has been deprecated in favor of fix_initial, which has '
                             'the opposite meaning. If the user desires to input the initial '
                             'phase time from an exterior source, set input_initial=True.')
        else:
            self.time_options['fix_initial'] = fix_initial

        if opt_duration is not None:
            self.time_options['fix_duration'] = not opt_duration
            warn_deprecation('opt_duration has been deprecated in favor of fix_duration, which has '
                             'the opposite meaning. If the user desires to input the phase '
                             'duration from an exterior source, set input_duration=True.')
        else:
            self.time_options['fix_duration'] = fix_duration

        # Don't allow the user to provide desvar options if the time is not a desvar or is input.
        if input_initial and self.time_options['fix_initial']:
            warnings.warn('Phase "{0}" initial time is an externally-connected input, '
                          'therefore fix_initial has no effect.'.format(self.name), RuntimeWarning)
        elif input_initial or self.time_options['fix_initial']:
            illegal_options = []
            if initial_bounds != (None, None):
                illegal_options.append('initial_bounds')
            if initial_scaler is not None:
                illegal_options.append('initial_scaler')
            if initial_adder is not None:
                illegal_options.append('initial_adder')
            if initial_ref is not None:
                illegal_options.append('initial_ref')
            if initial_ref0 is not None:
                illegal_options.append('initial_ref0')
            if illegal_options:
                reason = 'input_initial=True' if input_initial else 'fix_initial=True'
                msg = 'Phase time options have no effect because {2} for phase ' \
                      '"{0}": {1}'.format(self.name, ', '.join(illegal_options), reason)
                warnings.warn(msg, RuntimeWarning)

        if input_duration and self.time_options['fix_duration']:
            warnings.warn('Phase "{0}" time duration is an externally-connected input, '
                          'therefore fix_duration has no effect.'.format(self.name),
                          RuntimeWarning)
        elif input_duration or self.time_options['fix_duration']:
            illegal_options = []
            if duration_bounds != (None, None):
                illegal_options.append('duration_bounds')
            if duration_scaler is not None:
                illegal_options.append('duration_scaler')
            if duration_adder is not None:
                illegal_options.append('duration_adder')
            if duration_ref is not None:
                illegal_options.append('duration_ref')
            if duration_ref0 is not None:
                illegal_options.append('duration_ref0')
            if illegal_options:
                reason = 'input_duration=True' if input_duration else 'fix_duration=True'
                msg = 'Phase time options have no effect because {2} for phase ' \
                      '"{0}": {1}'.format(self.name, ', '.join(illegal_options), reason)
                warnings.warn(msg, RuntimeWarning)

        self.time_options['input_initial'] = input_initial
        self.time_options['initial'] = initial
        self.time_options['initial_bounds'] = initial_bounds
        self.time_options['initial_scaler'] = initial_scaler
        self.time_options['initial_adder'] = initial_adder
        self.time_options['initial_ref'] = initial_ref
        self.time_options['initial_ref0'] = initial_ref0

        self.time_options['input_duration'] = input_duration
        self.time_options['duration'] = duration
        self.time_options['duration_bounds'] = duration_bounds
        self.time_options['duration_scaler'] = duration_scaler
        self.time_options['duration_adder'] = duration_adder
        self.time_options['duration_ref'] = duration_ref
        self.time_options['duration_ref0'] = duration_ref0

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
            'control_rate2', 'design_parameter', 'input_parameter', or 'ode'.

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
        elif var in self.design_parameter_options:
            return 'design_parameter'
        elif var in self.input_parameter_options:
            return 'input_parameter'
        elif var.endswith('_rate'):
            if var[:-5] in self.control_options:
                return 'control_rate'
        elif var.endswith('_rate2'):
            if var[:-6] in self.control_options:
                return 'control_rate2'
        else:
            return 'ode'

    def setup(self):
        transcription_order = self.options['transcription_order']

        if np.any(np.asarray(transcription_order) < 3):
            raise ValueError('Given transcription order ({0}) is less than '
                             'the minimum allowed value (3)'.format(transcription_order))

        self._time_extents = self._setup_time()

        # Declare control_rate comp to which we'll connect controls and parameters.
        if self.control_options:
            ctrl_rate_comp = ControlInterpComp(control_options=self.control_options,
                                               time_units=self.time_options['units'],
                                               grid_data=self.grid_data)
            self._setup_controls()

            self.add_subsystem('control_interp_comp',
                               subsys=ctrl_rate_comp,
                               promotes_inputs=['controls:*'],
                               promotes_outputs=['control_rates:*'])
            self.connect('time.dt_dstau', 'control_interp_comp.dt_dstau')

        if self.input_parameter_options:
            self._setup_input_parameters()

        if self.design_parameter_options:
            self._setup_design_parameters()

        self._setup_rhs()
        self._setup_defects()
        self._setup_states()

        self._setup_endpoint_conditions()
        self._setup_boundary_constraints('initial')
        self._setup_boundary_constraints('final')
        self._setup_path_constraints()

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
                indep.add_output(var, val=1.0, units=time_units)
            self.add_subsystem('time_extents', indep, promotes_outputs=['*'])
            comps += ['time_extents']

        time_comp = TimeComp(grid_data=grid_data, units=time_units)
        self.add_subsystem('time', time_comp, promotes_outputs=['time', 'time_phase'],
                           promotes_inputs=externals)

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
        return comps

    def _setup_controls(self):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        opt_controls = [name for (name, opts) in iteritems(self.control_options) if opts['opt']]

        num_opt_controls = len(opt_controls)

        grid_data = self.grid_data

        if num_opt_controls > 0:
            indep = self.add_subsystem('indep_controls', subsys=IndepVarComp(),
                                       promotes_outputs=['*'])

        num_dynamic_controls = 0

        for name, options in iteritems(self.control_options):
            if options['opt']:
                num_dynamic_controls = num_dynamic_controls + 1
                num_input_nodes = grid_data.subset_num_nodes['control_input']

                desvar_indices = list(range(self.grid_data.subset_num_nodes['control_input']))
                if options['fix_initial']:
                    desvar_indices.pop(0)
                if options['fix_final']:
                    desvar_indices.pop()

                if len(desvar_indices) > 0:
                    coerce_desvar = CoerceDesvar(grid_data.subset_num_nodes['control_disc'],
                                                 desvar_indices, options)

                    lb = -INF_BOUND if coerce_desvar('lower') is None else coerce_desvar('lower')
                    ub = INF_BOUND if coerce_desvar('upper') is None else coerce_desvar('upper')

                    self.add_design_var(name='controls:{0}'.format(name),
                                        lower=lb,
                                        upper=ub,
                                        scaler=coerce_desvar('scaler'),
                                        adder=coerce_desvar('adder'),
                                        ref0=coerce_desvar('ref0'),
                                        ref=coerce_desvar('ref'),
                                        indices=desvar_indices)

                indep.add_output(name='controls:{0}'.format(name),
                                 val=options['val'],
                                 shape=(num_input_nodes, np.prod(options['shape'])),
                                 units=options['units'])

        return num_dynamic_controls

    def _setup_design_parameters(self):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
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

        for name, options in iteritems(self.input_parameter_options):
            src_name = 'input_parameters:{0}_out'.format(name)

            for tgts, src_idxs in self._get_parameter_connections(options['target_param']):
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
        raise NotImplementedError()

    def _get_rate_source_path(self, state_name, nodes, **kwargs):
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
        raise NotImplementedError()

    def _setup_rhs(self):
        raise NotImplementedError()

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
            con_shape = options.get('shape', (1,))
            con_size = int(np.prod(con_shape))
            con_options['shape'] = shape if con_shape is None else con_shape
            con_options['units'] = units if con_units is None else con_units
            con_options['linear'] = linear

            # Build the correct src_indices regardless of shape
            if loc == 'initial':
                src_idxs = np.arange(con_size, dtype=int).reshape(con_shape)
            else:
                src_idxs = np.arange(-con_size, 0, dtype=int).reshape(con_shape)

            bc_comp._add_constraint(con_name, **con_options)

            self.connect(src,
                         '{0}_boundary_constraints.{0}_value_in:{1}'.format(loc, con_name),
                         src_indices=src_idxs,
                         flat_src_indices=True)

    def _setup_path_constraints(self):
        raise NotImplementedError('_setup_path_constraints has not been implemented '
                                  'for this phase type')

    def get_values(self, var, nodes=None, units=None):
        """
        Retrieve the values of the given variable at the given
        subset of nodes.

        Parameters
        ----------
        var : str
            The variable whose values are to be returned.  This may be
            the name 'time', the name of a state, control, or parameter,
            or the path to a variable in the ODE system of the phase.
        nodes : str or None
            The name of the node subset or None (default).
        units : str or None
            The units in which the returned values are to be provided.

        Returns
        -------
        ndarray
            An array of the values at the requested node subset.  The
            node index is the first dimension of the ndarray.
        """
        raise NotImplementedError('get_values has not been implemented for this class.')

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
