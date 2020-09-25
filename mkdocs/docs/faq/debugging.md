# How can I debug models when things go wrong?

Dymos allow the user to build complex optimization models that include dynamic behavior.
Managing that complexity can be a challenge as models grow larger.
In this section we'll talk about some tools that can help when things are going as expected.

## Testing

If you look at the dymos source code, a considerable portion of it is used for testing.
We strongly recommend that you develop tests of your models, from testing that the most basic components work as expected, to testing integrated systems with nonlinear solvers.
In most cases these tests consist of these steps:

1. Instantiate an OpenMDAO Problem
2. Add your model.
3. Setup the problem.
4. Set the model inputs.
5. Call `run_model()`
6. Check the outputs against known values.
7. Run `problem.check_partials()` to verify that the analytic partials are reasonably close to finite-difference or complex-step results.

For example, the tests for the `kappa_comp` in the minimum time-to-climb model looks like this:

=== "test_kappa_comp.py"
{{ inline_source('dymos.examples.min_time_climb.aero.test.test_kappa_comp',
include_def=True,  
include_docstring=True,
indent_level=0)
}}

This consists of two separate tests: one that tests results, and one that tests the partials against finite-differencing.
OpenMDAO includes a useful `assert_check_partials` method that can be used to programmatically verify accurate partials in automated testing.

## The N2 Viewer

When complex models don't output the correct value and the compute method has been double-checked, an incorrect or non-existent connection is frequently to blame.
The goto tool for checking to see if models are correctly connected is [OpenMDAO's N-squared (N2) viewer](http://openmdao.org/twodocs/versions/latest/features/model_visualization/n2_details.html).
This tool provides information about how models are connected and lets the user know when inputs aren't connected to an output as expected.

It can be invoked from a run script using

```python
om.n2(problem.model)
```

or from the command line using

```bash
openmdao n2 file.py
```

where file.py is the file that contains an instantiated OpenMDAO Problem.

## An example of an N2 of a Dymos model

Consider the steady-flight aircraft example.
The ODE in this problem contains a nonlinear solver that determines the thrust and angle-of-attack required to achieve the given flight profile.
Dymos itself consists of a lot of components used to provide the transcription.
In general, Dymos users will want to zoom in on the ODE to see the connections in their model.


