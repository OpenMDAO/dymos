import numpy as np
from six import iteritems

from dymos.glm.ozone.utils.misc import _get_class
from dymos.glm.ozone.methods_list import get_method


def ODEIntegrator(ode_class, formulation, method_name,
        initial_conditions=None, dynamic_parameters=None,
        initial_time=None, final_time=None, normalized_times=None, times=None,
        ode_init_kwargs=None, **kwargs):
    """
    Create and return an OpenMDAO group containing the ODE integrator.

    The returned group can be the model or a part of a larger model in OpenMDAO.

    Parameters
    ----------
    ode_class
        The OpenMDAO system class representing the 'f' in dy_dt = f(t, y, x).
    formulation : str
        Formulation for solving the ODE: 'time-marching', 'solver-based', or 'optimizer-based'.
    method_name : str
        The time integration method. The list of methods can be found in the documentation.
    initial_conditions : dict or None
        Optional dictionary of initial condition values keyed by state name.
        If not given here, it must be connected from outside the integrator group.
    dynamic_parameters : dict or None
        Optional dictionary of static parameter values keyed by parameter name.
        If not given here, it must be connected from outside the integrator group.
    initial_time : float or None
        Only required if times is not given and not connected from outside the integrator group.
    final_time : float or None
        Only required if times is not given and not connected from outside the integrator group.
    normalized_times : np.ndarray[:]
        Vector of times given if initial & final times are provided or connected from the outside.
        If given, this vector is normalized to the [0, 1] interval.
        Not necessary if times is provided.
    times : np.ndarray[:]
        Vector of times required if initial time, final time, and normalized_times are not given.

    Returns
    -------
    Group
        The OpenMDAO Group instance representing the requested integrator.
    """
    ode_options = ode_class.ode_options
    method = get_method(method_name)
    explicit = method.explicit
    integrator_class = get_integrator(formulation, explicit)

    # ------------------------------------------------------------------------------------
    # time-related option
    assert normalized_times is not None or times is not None, \
        'Either normalized_times or times must be provided'

    if normalized_times is not None:
        assert isinstance(normalized_times, np.ndarray) and len(normalized_times.shape) == 1, \
            'normalized_times must be a 1-D array'

    if times is not None:
        assert isinstance(times, np.ndarray) and len(times.shape) == 1, \
            'times must be a 1-D array'

        assert initial_time is None and final_time is None and normalized_times is None, \
            'If times is provided: initial_time, final_time, and normalized_times cannot be'

        initial_time = times[0]
        final_time = times[-1]
        normalized_times = (times - times[0]) / (times[-1] - times[0])

    # ------------------------------------------------------------------------------------
    # Ensure that all initial_conditions are valid
    if initial_conditions is not None:
        for state_name, value in iteritems(initial_conditions):
            assert state_name in ode_options._states, \
                'Initial condition (%s) was not declared in ODE Options' % state_name

            assert isinstance(value, np.ndarray) or np.isscalar(value), \
                'The initial condition for state %s must be an ndarray or a scalar' % state_name

            assert np.atleast_1d(value).shape == ode_options._states[state_name]['shape'], \
                'The initial condition for state %s has the wrong shape' % state_name

            initial_conditions[state_name] = np.atleast_1d(value)

    # ------------------------------------------------------------------------------------
    # Ensure that all dynamic parameters are valid
    if dynamic_parameters is not None:
        num_times = len(normalized_times)

        for parameter_name, value in iteritems(dynamic_parameters):
            assert parameter_name in ode_options._dynamic_parameters, \
                'Dynamic parameter (%s) was not declared in ODE Options' % parameter_name

            assert isinstance(value, np.ndarray), \
                'Dynamic parameter %s must be an ndarray' % parameter_name

            shape = ode_options._dynamic_parameters[parameter_name]['shape']
            assert value.shape == (num_times,) + shape, \
                'Dynamic parameter %s has the wrong shape' % state_name

    # ------------------------------------------------------------------------------------

    if formulation == 'optimizer-based' or formulation == 'solver-based':
        kwargs['formulation'] = formulation

    integrator = integrator_class(ode_class=ode_class, method=method,
        initial_conditions=initial_conditions, dynamic_parameters=dynamic_parameters,
        initial_time=initial_time, final_time=final_time, normalized_times=normalized_times,
        all_norm_times=normalized_times, ode_init_kwargs=ode_init_kwargs,
        **kwargs)

    return integrator


def get_integrator(formulation, explicit):
    from dymos.glm.ozone.integrators.explicit_tm_integrator import ExplicitTMIntegrator
    from dymos.glm.ozone.integrators.implicit_tm_integrator import ImplicitTMIntegrator
    from dymos.glm.ozone.integrators.vectorized_integrator import VectorizedIntegrator

    integrator_classes = {
        'optimizer-based': VectorizedIntegrator,
        'solver-based': VectorizedIntegrator,
        'time-marching': ExplicitTMIntegrator if explicit else ImplicitTMIntegrator,
    }
    return _get_class(formulation, integrator_classes, 'Integrator')
