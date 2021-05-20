# How do I add an ODE output to the timeseries outputs?

The timeseries object in Dymos provides a transcription-independent way to get timeseries output of a variable in a phase.
By default, these timeseries outputs include Dymos phase variables (times, states, controls, and parameters).
Often, there will be some other intermediate or auxiliary output calculations in the ODE that we want to track over time.
These can be added to the timeseries outputs using the `add_timeseries_output` method on Phase.

Multiple timeseries outputs can be added at one time by matching a glob pattern.
For instance, to add all outputs of the ODE to the timeseries, one can use '*' as the


See the [Timeseries documentation](../features/phases/timeseries.md) for more information.


The [commercial aircraft example](../examples/commercial_aircraft/commercial_aircraft.md) uses the `add_timeseries_output` method to add the lift and drag coefficients to the timeseries outputs.
