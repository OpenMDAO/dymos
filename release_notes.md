# Dymos Release Notes

## 0.16.0
2020-10-23

This is a long-overdue release of Dymos that features several useful new features.
By taking advantage of recent changes to the OpenMDAO setup process, we can use introspection to determine a lot of things automatically now.
For instance, the user will no longer be required to specify the shape or default units of timeseries outputs or constraints (although they can override units if they choose).

Also on the topic of timeseries, dymos now supports glob patterns when adding ODE outputs to the timeseries.
Adding all 100 outputs of a large ODE to the timeseries output can now be accomplished in a single line:

```
phase.add_timeseries_output('*')
```

Introspection is also used for states, time, and controls.
States will get their default units and shape from the associated state rate variable now.
Controls, Polynomial Controls, and Parameters will now get their default units and shapes from their targets.

With the advent of automatic IndepVarComps in OpenMDAO, there no longer needs to be a distinction between InputParameters and DesignParameters.
From here on out, users just add `parameters` to a Trajectory or Phase.
A user can choose to make them design variables (in which case they perform as DesignParameters), or not.
Connecting an external variable to the parameter makes it function as a InputParameter.

From here on, we plan on making releases of Dymos roughly once per month.

* [__Enhancement__] Phase linkage enhancments.  Any outputs can be linked across phases.  Added a more general and powerful `add_linkage_constraint` method, but the simpler `link_phases` method remains in place for simple continuity. #422
* [__Enhancement__] Time, controls, and polynomial control units. #412
* [__Bug Fix__] AnalysisError is now raised if scipy.integrate.solve_ivp fails during simulation. #400
* [__Enhancement__] States now pull their default units and shapes from the rate source. #398
* [__Enhancement__] Phase method `add_timeseries_output` now supports wildcards to allow multiple timeseries outputs to be added at once. #387, #399
* [__Enhancement__] Timeseries outputs automatically detect shape and default units. #380
* [__Enhancement__] Added ph-adaptive refinement method that is capable of shrinking the grid. #379
* [__Enhancement__] Control targets will automatically be set to a top-level input of the ODE, if present. #373
  * [__Enhancement__] API Change:  `design_parameters` and `input_parameters` are now just `parameters`. Old functionality deprecated. #365
* [__Enhancement__] State targets will automatically be set to a top-level input of the ODE, if present. #356
* [__Enhancement__] Water-powered rocket MDO example added. #343
* [__Docs__] Documentation is now handled via mkdocs instead of sphinx. #337, #332
* [__Enhancement__] State rates are included in timeseries outputs by default. #329
* [__Enhancement__] By default, run_problem will use a problem recorder to record only the 'final' solution after an optimization. #328
* [__Enhancement__] Move most setup functionality to configure to allow more introspection changes. #327
* [__Bug Fix__] Switch to use preferred `assert_near_equal` method instead of `assert_rel_error` from OpenMDAO. #320
* [__Bug Fix__] Fix for simulate encountering errors when time options `input_initial` or `input_duration` were True. #317
* [__Enhancement__] Include examples using IPOPT via pyoptsparse. #311
* [__Enhancement__] Parameters may now be selectively omitted from the timeseries outputs, but are included by default. #305
* [__Enhancement__] Added a test of the distributed ODE capability. #304
* [__Bug Fix__] Fixed a bug in which shaped input parameters were breaking simulate. #301
* [__Bug Fix__] Sort linkage orders to make convergence more repeatable. #298
* [__Enhancement__] Replace `load_case` with reinterpolate solution.
* [__Bug Fix__] Fixed a bug involving GaussLobatto transcriptions `get_rate_source` method. #334

## 0.15.0
2020-02-12

* [__Bug Fix__] Phase Linkage units now checks all optimal control variables from the first phase in the linkage for units. Previously units for things that were neither states, time, nor controls were not being assigned.
* [__Enhancement__] Removed Python 2 support.
* [__Enhancement__] Removed deprecated ODE Decorators.
* [__Enhancement__] Added ability to subclass Phase with an assigned ODE and default state variable options.
* [__Docs__] Added docs on _subclassing phases_.
* [__Enhancement__] Automated grid refinement is now available via the `dymos.run_problem` function.
* [__Enhancement__] Grid data is now available upon instantiation of the Transcription instead of being deferred to setup time.
* [__Bug Fix__] The user now gets a meaningful message if Phase.interpolate is called too soon.
* [__Bug Fix__] State rates are now correctly passed through the interleave component that provides timeseries outputs for Gauss-Lobatto transcription.
* [__Enhancement__] Added hypersensitive example problem.
* [__Docs__] Documentation added for grid refinement.
* [__Enhancement__] Deprecated the use of ODE decorators
* [__Enhancement__] Added shuttle reentry example problem

## 0.13.0
2019-07-18

* [__Enhancement__] Phase methods like `set_state_options` and `add_control` no longer use **kwargs in order to make them more IDE friendly.
* [__Enhancement__] Additional timeseries can for outputs can be added to a phase, with interpolation onto a new set of grid points. This enables the concept of tandem phases, where two different ODE's operating over the same time interval can be integrated on different grids. Fast state variables can be integrated on a dense grid while slower state variables are integrated on a more sparse grid, for performance. For an example see the tandem phase documentation in the feature docs.
* [__Enhancement__] ODE options can be specified at the phase level rather than in the ODE. This feature is experimental, but it allows one way of programmatically defining an ODE.
* [__Enhancement__] Changed the use of OpenMDAO to do `import openmdao.api as om` for consistency with the OpenMDAO documentation.
* [__Enhancement__] Dymos is now imported in the examples as import `dymos as dm`

