from __future__ import print_function, division, absolute_import

from collections import Iterable
from six import string_types
import numpy as np

from openmdao.utils.options_dictionary import OptionsDictionary


class ODEFunction(object):
    """
    Base class for required user-defined ode function.

    Define an ODE of the form y' = f(t, x, y).
    """

    def __init__(self, **kwargs):
        """
        Initialize class attributes.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments that will be passed to the initialize method.
        """
        self._system_class = None
        self._system_init_kwargs = {}

        time_options = OptionsDictionary()
        time_options.declare('targets', default=[], types=Iterable)
        time_options.declare('units', default=None, types=string_types, allow_none=True)

        self._time_options = time_options
        self._states = {}
        self._static_parameters = {}
        self._dynamic_parameters = {}

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

    def set_system(self, system_class, system_init_kwargs=None):
        """
        Set the OpenMDAO System that computes the ODE function.

        Parameters
        ----------
        system_class : System
            OpenMDAO Group or Component class defining our ODE.
        system_init_kwargs : dict or None
            Dictionary of kwargs that should be passed in when instantiating system_class.
        """
        self._system_class = system_class
        if system_init_kwargs is not None:
            self._system_init_kwargs = system_init_kwargs

    def declare_time(self, targets=None, units=None):
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
        if units is not None:
            self._time_options['units'] = units

    def declare_state(self, name, rate_source, targets=None, shape=None, units=None):
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

        options = OptionsDictionary()
        options.declare('name', types=string_types)
        options.declare('rate_source', types=string_types)
        options.declare('targets', default=[], types=Iterable)
        options.declare('shape', default=(1,), types=tuple)
        options.declare('units', default=None, types=string_types, allow_none=True)

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
            defined by the shape argument.
        """
        if dynamic:
            self._declare_dynamic_parameter(name, targets, shape=shape, units=units)
        else:
            self._declare_static_parameter(name, targets, shape=shape, units=units)

    def _declare_static_parameter(self, name, targets, shape=None, units=None):
        """
        Declare an input to the ODE.

        Parameters
        ----------
        name : str
            The name of the static parameter.
        targets : string_types or Iterable or None
            Paths to inputs in the ODE to which the incoming value of the static parameter
            needs to be connected.
        shape : int or tuple or None
            Shape of the parameter.
        units : str or None
            Units of the parameter.
        """
        if name in self._static_parameters:
            raise ValueError('static parameter {0} has already been declared.'.format(name))

        options = OptionsDictionary()
        options.declare('name', types=string_types)
        options.declare('targets', default=[], types=Iterable)
        options.declare('shape', default=(1,), types=tuple)
        options.declare('units', default=None, types=string_types, allow_none=True)

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

        self._static_parameters[name] = options

    def _declare_dynamic_parameter(self, name, targets, shape=None, units=None):
        """
        Declare an input to the ODE.

        Parameters
        ----------
        name : str
            The name of the dynamic parameter.
        targets : string_types or Iterable or None
            Paths to inputs in the ODE to which the incoming value of the dynamic parameter
            needs to be connected.
        shape : int or tuple or None
            Shape of the parameter.
        units : str or None
            Units of the parameter.
        """
        if name in self._dynamic_parameters:
            raise ValueError('Dynamic parameter {0} has already been declared.'.format(name))

        options = OptionsDictionary()
        options.declare('name', types=string_types)
        options.declare('targets', default=[], types=Iterable)
        options.declare('shape', default=(1,), types=tuple)
        options.declare('units', default=None, types=string_types, allow_none=True)

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

        self._dynamic_parameters[name] = options

    def get_test_parameters(self):
        """
        Optional method to provide default parameters; used for testing.

        Returns
        -------
        dict
            Dictionary of initial conditions keyed by state name.
        float
            Integration start time.
        float
            Integration end time.
        """
        pass

    def get_exact_solution(self, initial_conditions, t0, t):
        """
        Optional method to compute the exact solution at time t given initial conditions at t0.

        Parameters
        ----------
        initial_conditions : dict
            Dictionary of initial conditions keyed by state name.
        t0 : float
            Integration start time.
        t : float
            Time at which the exact solution is desired.
        """
        pass
