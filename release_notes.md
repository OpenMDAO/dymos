********************************
# Release Notes for Dymos 0.17.0

December 14, 2020

This release of Dymos adds a few important features:
- Dymos can now automatically generate plots of all timeseries outputs
- State rates may now be tagged in the ODE

There are also several bug fixes.

## Backwards Incompatible API Changes & Deprecations

* None

## Enhancements

* Dymos can now automatically generate plots of all timeseries outputs. [#469](https://github.com/OpenMDAO/dymos/pull/469)
* States can now be discovered automatically by tagging their rate variables in the ODE. [#477](https://github.com/OpenMDAO/dymos/pull/477) [#481](https://github.com/OpenMDAO/dymos/pull/481) [#484](https://github.com/OpenMDAO/dymos/pull/484)

## Bug Fixes

* Fixed issue where polynomial controls were not able to be linked across phases. [#462](https://github.com/OpenMDAO/dymos/pull/462)
* Fixed a bug that was preventing parameters from being state rate sources. [#466](https://github.com/OpenMDAO/dymos/pull/466)
* Users may now specify parameter shapes as integers or iterables other than tuples. [#467](https://github.com/OpenMDAO/dymos/pull/467)

## Miscellaneous

* Fix for apache license string in setup.py classifiers since it was not recognized by PyPI. [#458](https://github.com/OpenMDAO/dymos/pull/458)
* Added a long description for PyPI. [#460](https://github.com/OpenMDAO/dymos/pull/460)
* Some documentation cleanup for the JOSS review  [#474](https://github.com/OpenMDAO/dymos/pull/474) [#488](https://github.com/OpenMDAO/dymos/pull/488)
* Dropped the dependency on the parameterized package with the intent to utilize subTest in the future.  [#479](https://github.com/OpenMDAO/dymos/pull/479)


********************************
# Release Notes for Dymos 0.16.1

November 16, 2020

This release of Dymos fixes an issue that now allows portions of an array
output to be connected to a parameter.

This version works with OpenMDAO 3.3.0 but version 3.4.1 offers some
improved handling of parameters.

## Backwards Incompatible API Changes & Deprecations

* Parameter shapes in 0.16.1 were stored as (1,) + the shape of the parameter.  Now they are shaped as expected. [#444](https://github.com/OpenMDAO/dymos/pull/444)

## Enhancements

* State shapes and units, if not explicitly given, are now pulled from targets (if present and uniquely defined), or from the rate source variable. [#449](https://github.com/OpenMDAO/dymos/pull/449)

## Bug Fixes

* Fixes a bug where user-defined shapes of states were colliding with those found during introspection, and other state introspection updates. [#449](https://github.com/OpenMDAO/dymos/pull/449)

* Fixes a bug that prevented the use of numpy arrays as a boundary constraints [#450](https://github.com/OpenMDAO/dymos/pull/450)

## Miscellaneous

* Added test to verify functionality of OpenMDAO 3.4.1 that allows the final value of a control (or a partial portion of any output) to be connected to a parameter. [#445](https://github.com/OpenMDAO/dymos/pull/445)

********************************
# Release Notes for Dymos 0.16.0

October 23, 2020

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

## Backwards Incompatible API Changes & Deprecations:

- `add_input_parameter` and `add_design_parameter` are **deprecated** and replaced with `add_parameter` [#365](https://github.com/OpenMDAO/dymos/pull/365)
- The use of two-character location strings ('--' and '++') are replaced by 'initial' and 'final' when linking phases. [#427](https://github.com/OpenMDAO/dymos/pull/427)
- The endpoint conditions component, used to pull out initial or final values of a variable in a phase, have been removed.  Values should instead by pulled from the timeseries. [#427](https://github.com/OpenMDAO/dymos/pull/427)
- The results of simulate are now automatically recorded to a database recorder 'dymos_simulation.db' and stored in a case named 'final'.  Use `CaseReader('dymos_simulation.db').get_case('final')` to access the case. instead of the previous behavior `CaseReader('dymos_simulation.db').get_case(-1)` [#437](https://github.com/OpenMDAO/dymos/pull/437)
- The RungeKutta transcription is **deprecated**. It is functionally equivalent to the GaussLobatto transcription with an order of 3, but since it's signficantly different under the hood it was being a drag on development.

One note on another backwards incompatibility.
Since we're using some newer OpenMDAO capabilities for the way in which parameters are handled, it is currently not possible to connect an upstream output to a parameter in dymos _when src_indices are used_.
We plan on fixing this issue with an upcoming OpenMDAO release.
If you rely on this capability, we recommend waiting for that update before using this version of Dymos, or using an intermediate "pass-through" component to pull the correct index from the original output and then passing _that_ as the value of a dymos parameter.

## Enhancements:

* Dymos components are no longer listed in check_partials output by default.  Use `dymos.options['include_check_partials'] = True` to override this for debugging.  [#438](https://github.com/OpenMDAO/dymos/pull/438)
* Any outputs can now be linked across phases.  Added a more general and powerful `add_linkage_constraint` method, but the simpler `link_phases` method remains in place for simple continuity. [#422](https://github.com/OpenMDAO/dymos/pull/422)
* Time, controls, and polynomial control units are now determined from targets. [#412](https://github.com/OpenMDAO/dymos/pull/412)
* States now pull their default units and shapes from the rate source. [#398](https://github.com/OpenMDAO/dymos/pull/398)
* Phase method `add_timeseries_output` now supports wildcards to allow multiple timeseries outputs to be added at once. [#387](https://github.com/OpenMDAO/dymos/pull/387), #387
* Timeseries outputs automatically detect shape and default units. [#380](https://github.com/OpenMDAO/dymos/pull/380)
* Added ph-adaptive refinement method that is capable of shrinking the grid. [#379](https://github.com/OpenMDAO/dymos/pull/379)
* Control targets will automatically be set to a top-level input of the ODE, if present. [#373](https://github.com/OpenMDAO/dymos/pull/373)
* State targets will automatically be set to a top-level input of the ODE, if present. [#356](https://github.com/OpenMDAO/dymos/pull/356)
* Water-powered rocket MDO example added. [#343](https://github.com/OpenMDAO/dymos/pull/343)
* State rates are included in timeseries outputs by default. [#329](https://github.com/OpenMDAO/dymos/pull/329)
* By default, run_problem will use a problem recorder to record only the 'final' solution after an optimization. [#328](https://github.com/OpenMDAO/dymos/pull/328)
* Move most setup functionality to configure to allow more introspection changes. [#327](https://github.com/OpenMDAO/dymos/pull/327)
* Include examples using IPOPT via pyoptsparse. [#311](https://github.com/OpenMDAO/dymos/pull/311)
* Parameters may now be selectively omitted from the timeseries outputs, but are included by default. [#305](https://github.com/OpenMDAO/dymos/pull/305)
* Added a test of the distributed ODE capability. [#304](https://github.com/OpenMDAO/dymos/pull/304)
* Replace `load_case` with `reinterpolate_solution`. [#296](https://github.com/OpenMDAO/dymos/pull/296)

## Bug Fixes:

* Fixed a bug that was causing matrix-shaped states to have connection errors. [#441](https://github.com/OpenMDAO/dymos/pull/441)
* Fixed a bug where simulate failed when both standard and polynomial controls were present. [#434](https://github.com/OpenMDAO/dymos/pull/434)
* Fixed an issue where parameters are promoted twice in some cases. This will cause errors in an upcoming versions of OpenMDAO. [#429](https://github.com/OpenMDAO/dymos/pull/429)
* Fixed a bug that was causing matrix-shaped states to have connection errors. [#441](https://github.com/OpenMDAO/dymos/pull/441)
* AnalysisError is now raised if scipy.integrate.solve_ivp fails during simulation. [#400](https://github.com/OpenMDAO/dymos/pull/400)
* Fixed a bug involving GaussLobatto transcriptions `get_rate_source` method. [#334](https://github.com/OpenMDAO/dymos/pull/334)
* Fix for simulate encountering errors when time options `input_initial` or `input_duration` were True. [#317](https://github.com/OpenMDAO/dymos/pull/317)
* Fixed a bug in which shaped input parameters were breaking simulate. [#301](https://github.com/OpenMDAO/dymos/pull/301)
* Sort linkage orders to make convergence more repeatable. [#298](https://github.com/OpenMDAO/dymos/pull/298)

## Miscellaneous:

* Switch to use preferred `assert_near_equal` method instead of `assert_rel_error` from OpenMDAO. [#320](https://github.com/OpenMDAO/dymos/pull/320)
* Documentation is now handled via mkdocs instead of sphinx. [#337](https://github.com/OpenMDAO/dymos/pull/337), [#332](https://github.com/OpenMDAO/dymos/pull/332)

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
