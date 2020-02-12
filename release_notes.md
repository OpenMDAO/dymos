# Dymos Release Notes

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

