# Timeseries Outputs

Different optimal control transcriptions work in different ways.
The Radau Pseudospectral transcription keeps a contiguous vector of state values at all nodes.
The Gauss Lobatto transcription keeps two separate continuous vectors; one at the discretization nodes and the other at the collocation nodes.
The RungeKutta transcription has a contiguous vector of values but only those values at segment endpoints are valid.
Retrieving a timeseries values of output values is thus transcription dependent.

In order to make obtaining the timeseries output of a phase easier, each phase provides a timeseries component which collects and outputs the appropriate timeseries data.
For the pseudospectral transcriptions, timeseries outputs are provided at all nodes.
For the RungeKutta transcription, timeseries outputs are provided at the segment endpoints.
By default, the timeseries output will include the following variables for every problem.

### Paths to timeseries outputs in Dymos

|Path                                                          | Description                                         |
|--------------------------------------------------------------|-----------------------------------------------------|
|``<phase path>.timeseries.time``                              |Current time value                                   |
|``<phase path>.timeseries.time_phase``                        |Current phase elapsed time                           |
|``<phase path>.timeseries.states:<x>``                        |Value of state variable named x                      |
|``<phase path>.timeseries.controls:<u>``                      |Value of control variable named u                    |
|``<phase path>.timeseries.control_rates:<u>_rate``            |Time derivative of control named u                   |
|``<phase path>.timeseries.control_rates:<u>_rate2``           |Second time derivative of control named u            |
|``<phase path>.timeseries.polynomial_controls:<p>``           |Value of polynomial control variable named u         |
|``<phase path>.timeseries.polynomial_control_rates:<p>_rate`` |Time derivative of polynomial control named u        |
|``<phase path>.timeseries.polynomial_control_rates:<p>_rate2``|Second time derivative of polynomial control named u |
|``<phase path>.timeseries.parameters:<d>``                    |Value of parameter named d                           |

## Adding additional timeseries outputs

In addition to these default values, any output of the ODE can be added to the timeseries output
using the ``add_timeseries_output`` method on Phase.  These outputs are available as
``<phase path>.timeseries.<output name>``.  A glob pattern can be used with ``add_timeseries_output``
to add multiple outputs to the timeseries simultaneously.  For instance, just passing '*' as the variable
name will add all dynamic outputs of the ODE to the timeseries.

Dymos will ignore any ODE oututs that are not sized such that the first dimension is the same as the
number of nodes in the ODE.  That is, if the output variable doesn't appear to be dynamic, it will not
be included in the timeseries outputs.

{{ api_doc('dymos.Phase.add_timeseries_output') }}

## Interpolated Timeseries Outputs

Sometimes a user may want to interpolate the results of a phase onto a different grid.  This is particularly
useful in the context of tandem phases.  Additional timeseries may be added to a phase using the
``add_timeseries`` method.  By default all timeseries will provide times, states, controls, and
parameters on the specified output grid.  Adding other variables is accomplished using the
``timeseries`` argument in the ``add_timeseries_output`` method.

{{ api_doc('dymos.Phase.add_timeseries') }}
