# How do I add an ODE output to the timeseries outputs?

The timeseries object in Dymos provides a transcription-independent way to get timeseries output of a variable in a phase.
By default, these timeseries outputs include Dymos phase variables (times, states, controls, and parameters).
Often, there will be some other intermediate or auxiliary output calculations in the ODE that we want to track over time.
These can be added to the timeseries outputs using the `add_timeseries_output` method on Phase.


See the [Phase API docs](../api/phase_api.md) for more information.


The [commercial aircraft example](../examples/commercial_aircraft/commercial_aircraft.md) uses the `add_timeseries_output` method to add the lift and drag coefficients to the timeseries outputs.


!!! info
    Currently the user is required to specify the units and shape of the outputs being added in the `add_timeseries_output` method.
    Future versions of OpenMDAO will allow us to infer this information automatically at setup time.