from __future__ import print_function, division, absolute_import

from collections import Iterable
from six import string_types
import numpy as np

from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.core.component import _valid_var_name


class _ODETimeOptionsDictionary(OptionsDictionary):
    """
    OptionsDictionary for Time at the ODE level.  Note this does not include things that affect
    design variables or defect constraints.  Those options are set at the Phase level.
    """
    def __init__(self, read_only=False):
        super(_ODETimeOptionsDictionary, self).__init__(read_only)
        self.declare('targets', default=[], types=Iterable, allow_none=True,
                     desc='Target path(s) for the time variable, relative to the top-level system.')
        self.declare('time_phase_targets', default=[], types=Iterable, allow_none=True,
                     desc='Target path(s) for the time_phase variable '
                          '(the current phase elapsed time), relative to the top-level system.')
        self.declare('t_initial_targets', default=[], types=Iterable, allow_none=True,
                     desc='Target path(s) for the t_initial variable relative to the top-level '
                          'system.')
        self.declare('t_duration_targets', default=[], types=Iterable, allow_none=True,
                     desc='Target path(s) for the t_duration variable relative to the top-level '
                          'system.')
        self.declare('units', default=None, types=string_types, allow_none=True,
                     desc='Units for time.')


class _ODEStateOptionsDictionary(OptionsDictionary):
    """
    OptionsDictionary for States at the ODE level.  Note this does not include things that affect
    design variables or defect constraints.  Those options are set at the Phase level.
    """
    def __init__(self, read_only=False):
        super(_ODEStateOptionsDictionary, self).__init__(read_only)
        self.declare('name', types=string_types, desc='The name of the state variable.')
        self.declare('rate_source', types=string_types,
                     desc='The path to the output in the system which is the time-derivative '
                          'of the state.')
        self.declare('targets', default=[], types=Iterable,
                     desc='Target paths for the state, relative to the top-level system.')
        self.declare('shape', default=(1,), types=tuple,
                     desc='The shape of the variable (ignoring the time-dimension along '
                          'the first axis).')
        self.declare('units', default=None, types=string_types, allow_none=True,
                     desc='The units of the parameter.')


class _ODEParameterOptionsDictionary(OptionsDictionary):
    """
    OptionsDictionary for parameters at the ODE level.  Note this does not include things that
    affect design variables or defect constraints.  Those options are set at the Phase level.
    """
    def __init__(self, read_only=False):
        super(_ODEParameterOptionsDictionary, self).__init__(read_only)
        self.declare('name', types=string_types, desc='The name of the parameter.')
        self.declare('targets', default=[], types=Iterable,
                     desc='Target paths for the parameter, relative to the top-level system.')
        self.declare('shape', default=(1,), types=tuple,
                     desc='The shape of the variable (ignoring the time-dimension along '
                          'the first axis).')
        self.declare('dynamic', default=True, types=bool,
                     desc='If True, the value of the shape of the parameter will '
                          'be (num_nodes, ...), allowing the variable to be used as either a '
                          'static or dynamic control.  This impacts the shape of the partial '
                          'derivatives matrix.  Unless a parameter is large and broadcasting a '
                          'value to each individual node would be inefficient, users should stick '
                          'to the default value of True.')
        self.declare('units', default=None, types=string_types, allow_none=True,
                     desc='The units of the parameter.')


class declare_time(object):
    """
    Class Decorator used to attach ODE time metadata to an OpenMDAO system.

    This decorator can be stacked with `declare_state` and `declare_parameter`
    to provide all necessary ODE metadata for system.
    """
    def __init__(self, targets=None, units=None, time_phase_targets=None, t_initial_targets=None,
                 t_duration_targets=None):
        self.targets = targets
        self.time_phase_targets = time_phase_targets
        self.t_initial_targets = t_initial_targets
        self.t_duration_targets = t_duration_targets
        self.units = units

    def __call__(self, system_class):
        if not hasattr(system_class, 'ode_options'):
            setattr(system_class, 'ode_options', ODEOptions())

        system_class.ode_options.declare_time(targets=self.targets, units=self.units,
                                              time_phase_targets=self.time_phase_targets,
                                              t_initial_targets=self.t_initial_targets,
                                              t_duration_targets=self.t_duration_targets)
        return system_class


class declare_state(object):
    """
    Class Decorator used to attach ODE state metadata to an OpenMDAO system.

    This decorator can be stacked with `declare_time` and `declare_parameter`
    to provide all necessary ODE metadata for system.
    """
    def __init__(self, name, rate_source, targets=None, shape=None, units=None):
        self.name = name
        self.rate_source = rate_source
        self.targets = targets
        self.shape = shape
        self.units = units

    def __call__(self, system_class):
        if not hasattr(system_class, 'ode_options'):
            setattr(system_class, 'ode_options', ODEOptions())

        system_class.ode_options.declare_state(name=self.name, rate_source=self.rate_source,
                                               targets=self.targets, shape=self.shape,
                                               units=self.units)
        return system_class


class declare_parameter(object):
    """
    Class Decorator used to attach ODE parameter metadata to an OpenMDAO system.

    This decorator can be stacked with `declare_time` and `declare_state`
    to provide all necessary ODE metadata for system.
    """
    def __init__(self, name, targets=None, shape=None, units=None, dynamic=True):
        self.name = name
        self.targets = targets
        self.shape = shape
        self.units = units
        self.dynamic = dynamic

    def __call__(self, system_class):
        if not hasattr(system_class, 'ode_options'):
            setattr(system_class, 'ode_options', ODEOptions())

        system_class.ode_options.declare_parameter(name=self.name, targets=self.targets,
                                                   shape=self.shape, units=self.units,
                                                   dynamic=self.dynamic)
        return system_class


class ODEOptions(object):
    """
    A container for ODE metadata which allow a System to be used as an ODE Function.
    """
    def __init__(self, time_options=None, state_options=None, parameter_options=None, **kwargs):
        """
        Initialize class attributes.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments that will be passed to the initialize method.
        """
        self._time_options = _ODETimeOptionsDictionary()

        self._states = {}
        self._parameters = {}
        self._target_paths = []

        if time_options:
            self.declare_time(**time_options)

        if state_options:
            for state_name in state_options:
                self.declare_state(state_name, **state_options[state_name])

        if parameter_options:
            for param_name in parameter_options:
                self.declare_parameter(param_name, **parameter_options[param_name])

        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        """
        Optional method that calls declare_time, declare_state, and/or declare_dynamic_parameter.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed in during instantiation.
        """
        pass

    def _check_targets(self, name, targets):
        """
        Check that the targets used for this variable have not already been
        used by another variable.

        Parameters
        ----------
        name : str
            The name of the state, parameter, or 'time'
        targets : Iterable
            The targets to which the state, parameter, or 'time' are connected.

        Raises
        ------
        ValueError
            If thet one or more of the targets for the variable have already been
            used by another variable.
        """
        if targets is None:
            return
        for var in targets:
            if var in self._target_paths:
                raise ValueError('{0} has a path "{1}" that has already been '
                                 'used as the target path of another '
                                 'variable.'.format(name, var))
        # If no issues have been found, extend the existing list of targets
        self._target_paths.extend(targets)

    def declare_time(self, targets=None, units=None, time_phase_targets=None,
                     t_initial_targets=None, t_duration_targets=None):
        """
        Specify the targets and units of time or the time-like variable.

        Parameters
        ----------
        targets : string_types or Iterable or None
            Targets for the time or time-like variable within the ODE, or None if no models
            are explicitly time-dependent. Default is None.
        units : str or None
            Units for the integration variable within the ODE. Default is None.
        """
        if isinstance(targets, string_types):
            self._time_options['targets'] = [targets]
        elif isinstance(targets, Iterable):
            self._time_options['targets'] = targets
        elif targets is not None:
            raise ValueError('targets must be of type string_types or Iterable or None')

        if isinstance(time_phase_targets, string_types):
            self._time_options['time_phase_targets'] = [time_phase_targets]
        elif isinstance(time_phase_targets, Iterable):
            self._time_options['time_phase_targets'] = time_phase_targets
        elif time_phase_targets is not None:
            raise ValueError('time_phase_targets must be of type string_types or Iterable or None')

        if isinstance(t_initial_targets, string_types):
            self._time_options['t_initial_targets'] = [t_initial_targets]
        elif isinstance(t_initial_targets, Iterable):
            self._time_options['t_initial_targets'] = t_initial_targets
        elif t_initial_targets is not None:
            raise ValueError('t_initial_targets must be of type string_types or Iterable or None')

        if isinstance(t_duration_targets, string_types):
            self._time_options['t_duration_targets'] = [t_duration_targets]
        elif isinstance(t_duration_targets, Iterable):
            self._time_options['t_duration_targets'] = t_duration_targets
        elif t_duration_targets is not None:
            raise ValueError('t_duration_targets must be of type string_types or Iterable or None')

        if units is not None:
            self._time_options['units'] = units

        self._check_targets('time', self._time_options['targets'])

    def declare_state(self, name, rate_source, targets=None, shape=None,
                      units=None):
        """
        Add an ODE state variable.

        Parameters
        ----------
        name : str
            The name of the state variable as seen by the driver. This variable will
            exist as an interface to the ODE.
        rate_source : str
            The path to the variable within the ODE which represents the derivative of
            the state variable w.r.t. the variable of integration.
        targets : string_types or Iterable or None
            Paths to inputs in the ODE to which the incoming value of the state variable
            needs to be connected.
        shape : int or tuple or None
            The shape of the variable to potentially be provided as a control.
        units : str or None
            Units of the variable.
        """
        if name in self._states:
            raise ValueError('State {0} has already been declared.'.format(name))
        if not _valid_var_name(name):
            raise NameError("'{0}' is not a valid OpenMDAO variable name.".format(name))

        options = _ODEStateOptionsDictionary()

        options['name'] = name
        options['rate_source'] = rate_source

        if isinstance(targets, string_types):
            options['targets'] = [targets]
        elif isinstance(targets, Iterable):
            options['targets'] = targets
        elif targets is not None:
            raise ValueError('targets must be of type string_types or Iterable or None')
        if np.isscalar(shape):
            options['shape'] = (shape,)
        elif isinstance(shape, Iterable):
            options['shape'] = tuple(shape)
        elif shape is not None:
            raise ValueError('shape must be of type int or Iterable or None')
        if units is not None:
            options['units'] = units

        self._check_targets(name, options['targets'])
        self._states[name] = options

    def declare_parameter(self, name, targets, shape=None, units=None, dynamic=True):
        """
        Declare an input to the ODE.

        Parameters
        ----------
        name : str
            The name of the parameter.
        targets : string_types or Iterable or None
            Paths to inputs in the ODE to which the incoming value of the parameter
            needs to be connected.
        shape : int or tuple or None
            Shape of the parameter.
        units : str or None
            Units of the parameter.
        dynamic : bool
            If True, the parameter has a different value at each time step (dynamic parameter);
            otherwise, the parameter has the same value at all time steps (static parameter).
            A dynamic parameter should have shape (num_nodes, ...) where ... is
            defined by the shape argument.  Since the Jacobian shape differs depending on whether
            or not it is dynamic, it is generally a good idea to declare parameters as dynamic
            unless there is no chance that they will need to be used as dynamic controls.
        """
        if name in self._parameters:
            raise ValueError('Parameter {0} has already been declared.'.format(name))
        if not _valid_var_name(name):
            raise NameError("'{0}' is not a valid OpenMDAO variable name.".format(name))

        options = _ODEParameterOptionsDictionary()

        options['name'] = name
        if isinstance(targets, string_types):
            options['targets'] = [targets]
        elif isinstance(targets, Iterable):
            options['targets'] = targets
        elif targets is not None:
            raise ValueError('targets must be of type string_types or Iterable or None')
        if np.isscalar(shape):
            options['shape'] = (shape,)
        elif isinstance(shape, Iterable):
            options['shape'] = tuple(shape)
        elif shape is not None:
            raise ValueError('shape must be of type int or Iterable or None')
        if units is not None:
            options['units'] = units
        options['dynamic'] = dynamic

        self._check_targets(name, options['targets'])
        self._parameters[name] = options

    def __str__(self):

        s = 'Time Options:\n    targets: {0}\n    units: {1}'.format(self._time_options['targets'],
                                                                     self._time_options['units'])
        if self._states:
            s += '\nState Options:'

        for state in sorted(self._states):
            options = self._states[state]
            s += '\n    {name}' \
                 '\n        rate_source: {rate_source}' \
                 '\n        targets: {targets}' \
                 '\n        shape: {shape}' \
                 '\n        units: {units}'.format(name=state, rate_source=options['rate_source'],
                                                   targets=options['targets'],
                                                   shape=options['shape'], units=options['units'])
        if self._parameters:
            s += '\nParameter Options:'

        for param in sorted(self._parameters):
            options = self._parameters[param]
            s += '\n    {name}' \
                 '\n        targets: {targets}' \
                 '\n        shape: {shape}' \
                 '\n        dynamic: True' \
                 '\n        units: {units}'.format(name=param, targets=options['targets'],
                                                   shape=options['shape'], units=options['units'])

        return s


class _ForDocs(object):  # pragma: no cover
    """
    This class is provided as a way to automatically display options dictionaries in the docs,
    since these option dictionaries typically don't exist in instantiated form in the code base.
    """

    def __init__(self):
        self.time_options = _ODETimeOptionsDictionary()
        self.state_options = _ODEStateOptionsDictionary()
        self.parameter_options = _ODEParameterOptionsDictionary()
